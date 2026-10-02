"""Run with python -m prototype.demo. All energy and latency are simulated."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .controller import Action, Controller, State


DISCLAIMER = (
    "SYNTHETIC DEMO: no LLM, quantization, GPU execution, or power measurements. "
    "Energy and latency are fictional, not research results."
)


def fraction(*values):
    digest = hashlib.sha256(repr(values).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class ToyModel:
    """A deterministic token source with simulated shallow-draft errors.

    This is not a transformer. Positions stand in for token/cache state; a real
    backend must implement partial-layer execution and KV-cache rollback.
    """

    words = "This toy example drafts candidate tokens and verifies them before committing output .".split()

    def __init__(self, seed=42, hard_after=48):
        self.seed = seed
        self.hard_after = hard_after

    def target_token(self, prefix):
        return self.words[len(prefix) % len(self.words)]

    def draft_token(self, prefix, exit_depth):
        easy = len(prefix) < self.hard_after
        probability = ({4: 0.93, 8: 0.99} if easy else {4: 0.12, 8: 0.28})[exit_depth]
        correct = self.target_token(prefix)
        if fraction(self.seed, tuple(prefix), exit_depth) < probability:
            return correct
        return "[rejected-draft]"

    def cycle(self, prefix, action, remaining):
        if remaining <= 0:
            return [], []
        if action.ordinary:
            return [self.target_token(prefix)], []
        if remaining <= action.draft_length:
            raise ValueError("Speculative cycle needs room for a correction or bonus")
        candidates = []
        for _ in range(action.draft_length):
            candidates.append(self.draft_token(prefix + candidates, action.exit_depth))
        committed, checks = [], []
        for candidate in candidates:
            correct = self.target_token(prefix + committed)
            checks.append(candidate == correct)
            if candidate != correct:
                # Drop the rejected token and all later speculative state.
                committed.append(correct)
                return committed, checks
            committed.append(candidate)
        committed.append(self.target_token(prefix + committed))
        return committed, checks


def load_actions(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("source") != "SYNTHETIC_DEMO_ONLY":
        raise ValueError("This toy runner only supports explicitly synthetic profiles")
    return [Action(**row) for row in profile["actions"]]


def run_policy(actions, policy, tokens=96, seed=42):
    if tokens <= 0:
        raise ValueError("Token budget must be positive")
    if policy not in ("ordinary", "fixed", "latency", "energy"):
        raise ValueError("Unknown policy")
    model = ToyModel(seed)
    adaptive = policy in ("energy", "latency")
    controller = Controller(actions, overhead_j=0.025 if adaptive else 0.0,
                            overhead_ms=0.15 if adaptive else 0.0)
    state = State(context_length=0, remaining_tokens=tokens)
    output, trace = [], []
    energy_total = latency_total = 0.0
    while len(output) < tokens:
        state.context_length = len(output)
        state.remaining_tokens = tokens - len(output)
        if policy in ("energy", "latency"):
            estimate, reason = controller.choose(state, policy)
        else:
            name = "ordinary" if policy == "ordinary" else "draft4x5"
            eligible = controller.estimates(state)
            estimate = next((row for row in eligible if row["action"].name == name),
                            next(row for row in eligible if row["action"].ordinary))
            reason = "fixed policy (ordinary near token limit)"
        action = estimate["action"]
        committed, checks = model.cycle(output, action, state.remaining_tokens)
        energy, latency = action.cycle_costs(len(output))
        # Fictional total costs include a small fictional controller overhead.
        # Real CPU/controller energy would require a suitable measurement scope.
        jitter = 0.98 + 0.04 * fraction(seed, len(output), action.name, "energy")
        energy = energy * jitter + controller.overhead_j
        latency += controller.overhead_ms
        trace.append({
            "context_length": len(output), "action": action.name,
            "reason": reason, "accepted_draft_tokens": sum(checks),
            "verified_draft_tokens": len(checks), "committed_tokens": len(committed),
            "acceptance_estimate": estimate["acceptance_estimate"],
            "predicted_energy_per_token": estimate["energy_per_token"],
            "simulated_energy_j": energy, "simulated_latency_ms": latency,
        })
        output.extend(committed)
        state.observe(action, checks)
        energy_total += energy
        latency_total += latency
    return {
        "policy": policy, "tokens": output, "cycles": len(trace),
        "simulated_energy_j": energy_total,
        "simulated_j_per_token": energy_total / tokens,
        "simulated_latency_ms": latency_total,
        "action_counts": dict(Counter(row["action"] for row in trace)),
        "trace": trace,
    }


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=DISCLAIMER)
    parser.add_argument("--tokens", type=positive_int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("synthetic_profile.json"))
    parser.add_argument("--output", type=Path, default=Path("prototype/results/demo.json"))
    args = parser.parse_args()
    actions = load_actions(args.profile)
    results = [run_policy(actions, policy, args.tokens, args.seed)
               for policy in ("ordinary", "fixed", "latency", "energy")]
    reference = results[0]["tokens"]
    for result in results:
        result["matches_ordinary"] = result["tokens"] == reference
    if not all(result["matches_ordinary"] for result in results):
        raise RuntimeError("Verification failed: output differs from ordinary decoding")
    report = {"disclaimer": DISCLAIMER, "seed": args.seed,
              "precision": "one fixed synthetic precision", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(DISCLAIMER)
    print("\nPolicy       Cycles  Simulated J/token  Simulated ms  Output matches")
    for result in results:
        print(f"{result['policy']:<12} {result['cycles']:>6}  "
              f"{result['simulated_j_per_token']:>17.3f}  "
              f"{result['simulated_latency_ms']:>12.1f}  {result['matches_ordinary']}")
    adaptive = results[-1]
    print("\nEnergy-controller decisions:", adaptive["action_counts"])
    previous = None
    for row in adaptive["trace"]:
        if row["action"] != previous:
            print(f"  Token {row['context_length']:>3}: {row['action']} - {row['reason']}")
            previous = row["action"]
    print("\nReport:", args.output.resolve())


if __name__ == "__main__":
    main()
