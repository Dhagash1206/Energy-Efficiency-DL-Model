"""Run the feasibility matrix; writes only below this package directory."""
import os
from pathlib import Path
from .paths import CACHE, HUB, TOKEN, ensure
ROOT = Path(__file__).resolve().parent
ensure()
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(HUB)
os.environ["HF_TOKEN_PATH"] = str(TOKEN)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import hashlib
import json
import platform
import random
import urllib.request
import torch
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from .engine import generate, native_greedy
from .measurement import Meter
from .report import render

MODELS = {"smollm": "HuggingFaceTB/SmolLM2-135M",
          "layerskip": "facebook/layerskip-llama3.2-1B"}


def get_model(name, device, precision, revision):
    if name == "tiny":
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
                    intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=256,
                    attn_implementation="eager")).eval()
        tokenizer, resolved = None, None
    else:
        from huggingface_hub import HfApi
        resolved = HfApi().model_info(MODELS[name], revision=revision).sha
        tokenizer = AutoTokenizer.from_pretrained(MODELS[name], revision=resolved,
                            cache_dir=str(HUB), trust_remote_code=False)
        model = LlamaForCausalLM.from_pretrained(MODELS[name], revision=resolved,
                    cache_dir=str(HUB), trust_remote_code=False,
                    torch_dtype=torch.float32, attn_implementation="eager").eval()
    if precision == "int8":
        if device != "cpu":
            raise ValueError("Initial INT8 implementation uses CPU dynamic linear operators")
        torch.ao.quantization.quantize_dynamic(model.model.layers, {torch.nn.Linear},
                                               dtype=torch.qint8, inplace=True)
    elif precision == "fp16":
        model.half()
    return model.to(device), tokenizer, resolved


def prompts_for(name, tokenizer, count):
    if name == "tiny":
        return [dict(id=f"fixture-{i}", ids=[1, 4+i, 6], source="synthetic_logic_fixture")
                for i in range(count)]
    # OpenAI's public MIT-licensed GSM8K; persist source revision and byte hash.
    request = urllib.request.Request(
        "https://api.github.com/repos/openai/grade-school-math/commits/master",
        headers={"User-Agent": "selective-feedback-pilot"})
    revision = json.load(urllib.request.urlopen(request, timeout=30))["sha"]
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{revision}/grade_school_math/data/test.jsonl"
    raw = urllib.request.urlopen(url, timeout=60).read()
    destination = ROOT / "data"
    destination.mkdir(exist_ok=True)
    (destination / "gsm8k-test.jsonl").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    entries = [json.loads(line) for line in raw.decode().splitlines()][:count]
    return [dict(id=f"gsm8k-test-{i}", source=url, sha256=digest,
                 ids=tokenizer.encode("Question: " + entry["question"] + "\nAnswer:",
                                      add_special_tokens=True)[:192])
            for i, entry in enumerate(entries)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tiny", "smollm", "layerskip"], default="tiny")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.prompts, args.tokens, args.repeats, args.threads) < 1:
        parser.error("counts must be positive")
    if args.model == "tiny" and args.prompts > 50:
        parser.error("tiny fixture supports at most 50 prompts")
    torch.set_num_threads(args.threads)
    print("Loading model...", flush=True)
    model, tokenizer, revision = get_model(args.model, args.device, args.precision, args.revision)
    print("Preparing prompts...", flush=True)
    prompts = prompts_for(args.model, tokenizer, args.prompts)
    eos = [tokenizer.eos_token_id] if tokenizer and tokenizer.eos_token_id is not None else []
    layers = len(model.model.layers)
    depths = sorted(set([max(1, layers//4), max(1, layers//2)]))
    cases = [(0, 0, "none")]
    cases += [(d, k, mode) for d in depths for k in (1, 3)
              for mode in ("none", "selected", "all", "periodic")]
    reference = {p["id"]: native_greedy(model, p["ids"], args.tokens, eos) for p in prompts}
    # Warm every execution/observation path without including it in timing.
    for depth, length, mode in cases:
        generate(model, prompts[0]["ids"], min(args.tokens, 5), depth, length,
                 depths, mode, eos=eos)
    jobs = [(p, case, repeat) for p in prompts for case in cases for repeat in range(args.repeats)]
    random.Random(args.seed).shuffle(jobs)
    meter, rows = Meter(model.device), []
    try:
        for job_index, (prompt, (depth, length, mode), repeat) in enumerate(jobs):
            if job_index % 10 == 0:
                print(f"Running case {job_index+1}/{len(jobs)}", flush=True)
            output, measured = meter.run(lambda: generate(
                model, prompt["ids"], args.tokens, depth, length, depths, mode, eos=eos))
            agreement = output["tokens"] == reference[prompt["id"]]
            rows.append(dict(prompt_id=prompt["id"], repeat=repeat, depth=depth,
                        draft_length=length, observation_mode=mode, agreement=agreement,
                        **measured, **output))
            if not agreement:
                break
    finally:
        meter.close()
    result = dict(status="correctness_passed" if all(r["agreement"] for r in rows) else "FAILED",
          model=MODELS.get(args.model, "random_untrained_tiny_llama"),
          model_revision=revision, arguments=vars(args), prompts=prompts,
          environment=dict(python=platform.python_version(), torch=torch.__version__,
                           transformers=transformers.__version__, cuda=torch.cuda.is_available()),
          exit_trained=args.model == "layerskip", research_energy_evidence=False,
          limitations=["Pilot only; no novel selector implemented.",
                       "SmolLM/tiny are mechanics checks, not exit-trained substitutes.",
                       "CPU energy unavailable; GPU counters exclude CPU controller energy.",
                       "Full traces add overhead in every mode; not optimized serving.",
                       "Single task and small sample do not establish generalization."],
          rows=rows)
    # Paired observation cost by identical prompt/configuration/repetition.
    lookup = {(r["prompt_id"], r["depth"], r["draft_length"], r["repeat"], r["observation_mode"]): r for r in rows}
    deltas = []
    for row in rows:
        if row["observation_mode"] == "none":
            continue
        base = lookup.get((row["prompt_id"], row["depth"], row["draft_length"], row["repeat"], "none"))
        if base:
            deltas.append(dict(prompt_id=row["prompt_id"], depth=row["depth"],
                          draft_length=row["draft_length"], mode=row["observation_mode"],
                          repeat=row["repeat"], latency_delta_s=row["latency_s"]-base["latency_s"],
                          energy_delta_j=row["energy_j"]-base["energy_j"]
                          if row["energy_j"] is not None and base["energy_j"] is not None else None))
    result["paired_observation_deltas"] = deltas
    result["model_memory_bytes"] = model.get_memory_footprint()
    result["expected_runs"] = len(jobs)
    path = ROOT / "results" / f"{args.model}-{args.precision}-pilot.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(render(result), encoding="utf-8")
    print(json.dumps(dict(report=str(path), status=result["status"], runs=len(rows),
                          energy_available=any(r["energy_j"] is not None for r in rows))))
    if result["status"] == "FAILED":
        raise SystemExit("Correctness failed; efficiency conclusions prohibited")

if __name__ == "__main__":
    main()

