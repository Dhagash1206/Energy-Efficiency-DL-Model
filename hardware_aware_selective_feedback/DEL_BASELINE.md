# DEL baseline and matched energy comparison

`run.py --include-del` adds `del` and `fixed_depth12` to the existing seven
policies. All nine execute the same held-out prompts and repeats in randomized
order, using the same model, precision, attention backend, and energy boundary.
Every output is checked against native greedy token IDs.

This is an **independent, greedy-only implementation of the DEL method on this
repository's engine**, not the authors' software or a reproduction of their
published speedups. Its results describe this port. Do not label a win against
this port as a win against the optimized upstream implementation.

## Algorithm and provenance

Reference: Zarch, Gao, Jiang, and Annavaram, [DEL: Context-Aware Dynamic Exit
Layer for Efficient Self-Speculative Decoding, COLM 2025](https://arxiv.org/abs/2504.05598v2).
The implementation follows the TPL objective, decayed acceptance/confidence
statistics, and dynamic draft stopping described in Sections 4.1--4.4.

The [public repository](https://github.com/hoenza/DEL) was inspected for
interpretation. At inspection it had no declared top-level license, so no DEL
source files were copied or vendored. The inspected controller blob was
`8c3a3c02968e963ad3ca91fa72ade726b6cc55b5`, and the generator blob was
`3e0e284bed3be71b456ab5d293f595975a8efa01`. Existing LayerSkip helper attribution
and its [license](vendor/LICENSE.LayerSkip) remain in place.

`del_policy.py` maintains one controller per request:

1. Prefill the model; collect intermediate states for the last 32 prompt
   positions. Compare shadow predictions with full-model greedy predictions.
2. Estimate each exit's acceptance from exponentially decayed match/mismatch
   counts (`decay=0.95`). Track confidence sums for both classes.
3. Choose the exit maximizing expected tokens divided by `L + depth * drafts`.
   Expected tokens are evaluated as `1 + alpha + ... + alpha**drafts`, avoiding
   the division-by-zero problem when acceptance is one.
4. Draft at that exit, stopping on low top-1 probability, EOS, or the draft/output
   budget. Verify with the remaining layers and roll back rejected KV entries.
5. Update shadow statistics only through the accepted prefix and correction
   position, truncated at EOS. Select again for the next round.

All intermediate depths `1..L-1` are eligible. The full-model cache is complete
at each round boundary, so changing depth does not require replaying the prefix.

## Explicit implementation choices

The paper, pseudocode, and released code are not identical. This port resolves
the differences as follows; it does not claim bit-for-bit upstream parity.

| Detail | Choice in this port |
|---|---|
| Sampling | Greedy only, matching the local experiment; no speculative sampling implementation |
| Initialization | Full prefill followed by a first full-model token; last 32 prompt positions initialize statistics |
| TPL candidate lengths | Inclusive `0..max_drafts`, as in the paper; upstream controller allocates an exclusive upper range |
| Length during drafting | Confidence stopping up to `max_drafts`, not a hard cap at the TPL-optimal length; TPL selects the depth |
| TPL chooses zero drafts | Execute an ordinary full-model round and refresh shadow statistics |
| Threshold crossing | Include the token that triggers stopping, then verify it, consistent with the released generator; paper pseudocode checks before appending |
| Missing match/mismatch class | Use threshold zero until both classes exist; no fabricated class mean or NaN propagation |
| Valid-context denominator | Count actual valid prediction positions, including the correction position; avoid index/count ambiguity |
| EOS and token limit | Include EOS consistently with local native greedy and enforce the remaining budget |
| LM-head allocation | Batch shadow states in chunks of 32 positions by default, bounding temporary vocabulary logits on small GPUs |
| Precision | Same model precision as every other policy; probabilities calculated in FP32 |

Chunking can change runtime and floating-point rounding. It is a memory-management
choice for this port, not an established optimization over upstream DEL. There is
no extra transformer pass for shadow collection, but LM-head projections,
confidence calculations, hooks, and controller work still have a measurable cost.

## Run

Use the environment whose versions you intend to record. The root `.venv` and
system `python` may differ; the report records actual PyTorch/Transformers versions.

```powershell
python -m unittest hardware_aware_selective_feedback.tests.test_del -v
python -m hardware_aware_selective_feedback.run --probe --device cuda
```

Smoke test, with its own output:

```powershell
python -m hardware_aware_selective_feedback.run --include-del --device cuda --objective energy --calibration 1 --test 1 --tokens 8 --repeats 1 --observer-pairs 1 --output hardware_aware_selective_feedback/results/del-comparison/smoke.json
```

Matched pilot comparison:

```powershell
python -m hardware_aware_selective_feedback.run --include-del --device cuda --objective energy --calibration 2 --test 2 --tokens 32 --repeats 2 --observer-pairs 3 --output hardware_aware_selective_feedback/results/del-comparison/pilot.json
```

For the previous larger workload, use `--calibration 4 --test 8 --tokens 64
--repeats 3 --observer-pairs 6`, and give it a new output filename. These counts
are per dataset. `--include-del` requires an explicit output path to avoid
accidentally replacing `latest.*`. DEL can take considerably longer because it
observes every intermediate exit.

Optional DEL parameters: `--del-max-drafts 18`, `--del-decay 0.95`, and
`--del-projection-chunk 32`. Changing them changes the baseline configuration;
report the values. Quantization, compilation, graph drafting, and adaptive-draft
fallback cannot be combined with this baseline comparison. FP16 is supported by
the shared engine but must pass greedy checks before any efficiency interpretation.

## Reading the comparison

JSON adds:

- `del_baseline`: method identity, configuration, and measurement caveats.
- `del_comparisons`: adaptive versus DEL and adaptive versus fixed depth 12.
- `source_sha256`: fingerprints of the implementation files used for the run.
- `del_cycles` on DEL rows: chosen depths, TPL choices, thresholds, actual draft
  lengths, accepted/committed tokens, and valid/excluded shadow counts.

DEL rows use `depth: null` / `draft_length: null` because these vary during the
request. `observed_exits` lists candidate exits. DEL's feedback counters cover
generation cycles; initialization feedback is not included in those counters.

Comparisons require complete, unique prompt/repeat pairs with identical valid
greedy output. If either policy mismatches, the entire corresponding comparison
is invalid, rather than presenting a favorable subset. A DEL mismatch does not
exclude an unrelated fixed action from the energy selector. The raw policy table
still displays measurements and mismatch counts for diagnosis.
The existing fixed-action mismatch diagnostic rejects dynamic DEL cases with an
explanation instead of incorrectly replaying them as a fixed configuration;
inspect `del_cycles` and rerun the recorded DEL configuration for those cases.

Positive reduction means adaptive used less cost than its baseline; negative
means it used more. Prompt-level paired differences average repeats first.
There is no confidence interval or statistical-significance claim.

**Measurement limitation:** the existing meter includes generation and its
internal work. DEL's within-request choices and all shadow projections are inside
that boundary. The existing energy selector's request-level selection and update
remain outside it. Shared calibration is reported separately and is not amortized
into any policy. These numbers therefore do not establish end-to-end net savings.
The current experiment also has different action spaces and feedback schedules;
it compares complete policies, not an isolated ablation of the energy objective.

The fixed-depth-12 comparator tests whether adaptation helps beyond the setting
the earlier energy controller selected most frequently. It is a predetermined
baseline informed by earlier results, not an independently validated optimum.

## Validation scope

Tests cover TPL extremes, decay, valid-prefix statistics, absent-class thresholds,
independent hidden-state shadow checks, changing depths and ordinary rounds,
rejection/cache rollback, EOS, short budgets, chunk sizes, hook cleanup, pair
validation, and an end-to-end tiny-model report. Passing these checks does not
replace validating every real-model measured output on the target device.
