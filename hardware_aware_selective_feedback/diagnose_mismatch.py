"""Diagnose the FP16 mismatch on gsm8k-test-880 (token 21) and
gsm8k-test-411 (token 93).

Runs each prompt in three modes and reports the top-2 logit margin at the
diverging token so we can tell whether the mismatch is a near-tie (fixable
with FP32 projection) or a structural error (different forward paths).

Usage:
    python -m hardware_aware_selective_feedback.diagnose_mismatch
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "layerskip" / "cache"
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
if (CACHE / "token").exists() and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN_PATH"] = str(CACHE / "token")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from hardware_aware_selective_feedback.run import load_model, load_prompts, REVISION
from hardware_aware_selective_feedback.engine import native_greedy, generate

# ── prompts that diverge ────────────────────────────────────────────────────
CASES = [
    {"prompt_id": "gsm8k-test-880",  "mismatch_token": 21,  "depth": 4, "draft": 1},
    {"prompt_id": "gsm8k-test-411",  "mismatch_token": 93,  "depth": 4, "draft": 1},
]
TOKEN_LIMIT = 100


def top2_margin_at(logits_fp32: torch.Tensor, pos: int) -> dict:
    """Return top-1 token, top-2 token, and the gap at position `pos`."""
    row = logits_fp32[pos]           # [vocab]
    vals, idxs = row.topk(2)
    return {
        "top1_id":  idxs[0].item(),
        "top2_id":  idxs[1].item(),
        "top1_val": vals[0].item(),
        "top2_val": vals[1].item(),
        "margin":   (vals[0] - vals[1]).item(),
    }


@torch.inference_mode()
def run_fp16_with_margins(model, prompt_ids: list, limit: int,
                           depth: int, draft: int, mismatch_pos: int,
                           eos) -> dict:
    """Run generate() with log_margins=True and return cycle data."""
    result = generate(
        model, prompt_ids, limit,
        depth=depth, drafts=draft,
        eos=eos, log_margins=True,
    )
    tokens = result["tokens"]

    # Find which cycle contains the mismatch token position.
    # Tokens are committed cumulatively; track which cycle emitted token[mismatch_pos].
    pos_cursor = 0
    target_cycle = None
    cycle_margin = None
    for cyc in result["cycles"]:
        n = len(cyc["committed"])
        if pos_cursor + n > mismatch_pos:
            target_cycle = cyc
            # Position within this cycle's target sequence.
            local = mismatch_pos - pos_cursor
            if cyc.get("top2_margin") is not None:
                cycle_margin = cyc["top2_margin"][local]
            break
        pos_cursor += n

    return {
        "tokens":          tokens,
        "mismatch_token_id": tokens[mismatch_pos] if len(tokens) > mismatch_pos else None,
        "cycle_margin_at_mismatch": cycle_margin,
        "cycles_total":    len(result["cycles"]),
    }


@torch.inference_mode()
def run_fp32_reference(model_fp16, prompt_ids: list, limit: int, eos) -> list:
    """Run native_greedy forcing FP32 logits at every step."""
    ids = torch.tensor([prompt_ids], device=model_fp16.device)
    tokens, cache = [], None
    eos_set = set(eos)
    for _ in range(limit):
        out = model_fp16(ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        # FP32 argmax — removes FP16 near-tie ambiguity.
        ids = out.logits[:, -1].float().argmax(-1).reshape(1, 1)
        tok = ids[0, 0].item()
        tokens.append(tok)
        if tok in eos_set:
            break
    return tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, tokenizer, rev = load_model(device, "fp16", attn="auto")
    eos = (tokenizer.eos_token_id,) if tokenizer.eos_token_id is not None else ()
    print(f"Model loaded  revision={rev}\n")

    # Load the same prompt set used in the sweep (seed=44).
    _, heldout, _ = load_prompts(1, 4, 44)
    prompt_map = {}
    for p in heldout:
        ids = tokenizer.encode(p["text"], add_special_tokens=True)[:192]
        prompt_map[p["id"]] = ids

    results = {}
    for case in CASES:
        pid   = case["prompt_id"]
        mpos  = case["mismatch_token"]
        depth = case["depth"]
        draft = case["draft"]

        if pid not in prompt_map:
            print(f"[SKIP] {pid} not in this seed's held-out set")
            continue

        prompt_ids = prompt_map[pid]
        print(f"{'─'*60}")
        print(f"Prompt: {pid}  |  mismatch at token {mpos}")
        print(f"Prompt length: {len(prompt_ids)} tokens")

        # 1. FP16 greedy reference (current native_greedy with float() fix).
        ref_fp16 = native_greedy(model, prompt_ids, TOKEN_LIMIT, eos)
        print(f"  FP16 greedy  token[{mpos}] = {ref_fp16[mpos] if len(ref_fp16) > mpos else 'EOS'}")

        # 2. FP16 LayerSkip with margin logging.
        ls_fp16  = run_fp16_with_margins(model, prompt_ids, TOKEN_LIMIT,
                                          depth, draft, mpos, eos)
        match_fp16 = (ls_fp16["tokens"][:TOKEN_LIMIT] == ref_fp16[:TOKEN_LIMIT])
        print(f"  LayerSkip FP16  token[{mpos}] = {ls_fp16['mismatch_token_id']}  "
              f"match={match_fp16}")
        if ls_fp16["cycle_margin_at_mismatch"] is not None:
            print(f"  Top-2 logit margin at token {mpos}: "
                  f"{ls_fp16['cycle_margin_at_mismatch']:.6f}")
        else:
            print(f"  (margin not captured at token {mpos})")

        # 3. FP32 reference — authoritative near-tie resolver.
        ref_fp32 = run_fp32_reference(model, prompt_ids, TOKEN_LIMIT, eos)
        print(f"  FP32 greedy  token[{mpos}] = {ref_fp32[mpos] if len(ref_fp32) > mpos else 'EOS'}")

        # 4. Agreement check with FP32 as ground truth.
        agree_fp16_vs_fp32 = (ref_fp16[:TOKEN_LIMIT] == ref_fp32[:TOKEN_LIMIT])
        agree_ls_vs_fp32   = (ls_fp16["tokens"][:TOKEN_LIMIT] == ref_fp32[:TOKEN_LIMIT])
        print(f"  FP16-greedy agrees with FP32: {agree_fp16_vs_fp32}")
        print(f"  LayerSkip   agrees with FP32: {agree_ls_vs_fp32}")

        results[pid] = {
            "mismatch_token": mpos,
            "ref_fp16_token": ref_fp16[mpos] if len(ref_fp16) > mpos else None,
            "ls_fp16_token":  ls_fp16["mismatch_token_id"],
            "ref_fp32_token": ref_fp32[mpos] if len(ref_fp32) > mpos else None,
            "margin":         ls_fp16["cycle_margin_at_mismatch"],
            "fp16_greedy_matches_fp32": agree_fp16_vs_fp32,
            "ls_matches_fp32":          agree_ls_vs_fp32,
        }
        print()

    print(f"{'═'*60}")
    print("Summary:")
    print(json.dumps(results, indent=2))

    # Verdict.
    print()
    for pid, r in results.items():
        margin = r["margin"]
        if margin is not None and abs(margin) < 0.05:
            verdict = f"NEAR-TIE (margin={margin:.4f}) — FP32 projection will fix this"
        elif r["ls_matches_fp32"]:
            verdict = "LayerSkip matches FP32 ground truth — FP16 greedy is the outlier"
        else:
            verdict = "Structural mismatch — different forward paths diverge beyond rounding"
        print(f"  {pid}: {verdict}")


if __name__ == "__main__":
    main()
