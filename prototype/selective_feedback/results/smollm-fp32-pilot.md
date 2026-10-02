# Mechanics pilot report

Model: HuggingFaceTB/SmolLM2-135M
Revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
Precision: fp32
Correctness: correctness_passed (102/102 runs agree)

**This is a mechanics check, not evidence of research energy savings.**
SmolLM is not exit-trained. CPU energy is unavailable. Timing includes prefill,
decoding, observation and trace processing; paired differences are noisy.

| Depth | Draft length | Observation | Runs | Median request seconds | Median joules |
|---|---|---|---|---|---|
| 0 | 0 | none | 6 | 2.324518 | unavailable |
| 7 | 1 | all | 6 | 2.996929 | unavailable |
| 7 | 1 | none | 6 | 3.069419 | unavailable |
| 7 | 1 | periodic | 6 | 3.451412 | unavailable |
| 7 | 1 | selected | 6 | 2.998875 | unavailable |
| 7 | 3 | all | 6 | 3.581788 | unavailable |
| 7 | 3 | none | 6 | 3.453147 | unavailable |
| 7 | 3 | periodic | 6 | 3.070241 | unavailable |
| 7 | 3 | selected | 6 | 2.769546 | unavailable |
| 15 | 1 | all | 6 | 3.496477 | unavailable |
| 15 | 1 | none | 6 | 3.175249 | unavailable |
| 15 | 1 | periodic | 6 | 3.812918 | unavailable |
| 15 | 1 | selected | 6 | 3.609546 | unavailable |
| 15 | 3 | all | 6 | 4.325520 | unavailable |
| 15 | 3 | none | 6 | 3.697614 | unavailable |
| 15 | 3 | periodic | 6 | 3.838869 | unavailable |
| 15 | 3 | selected | 6 | 4.241039 | unavailable |

## Feedback checks

Valid one-step observations: 402.
Excluded observations: 576; these are not counted as failures.

## Next decision

No go/no-go conclusion is justified from this smoke test.
Use an exit-trained model and repeated sensor-backed measurements before
deciding whether selective observation can repay its costs.
The selected mode observes the first available alternative exit; it is a
fixed baseline, not the proposed adaptive selector.

## Limits

- Pilot only; no novel selector implemented.
-s checks, not exit-trained substitutes. SmolLM/tiny are mechanic
- CPU energy unavailable; GPU counters exclude CPU controller energy.
- Full traces add overhead in every mode; not optimized serving.
- Single task and small sample do not establish generalization.

## Scope of the completed smoke run

The saved FP32 SmolLM run uses two candidate exits (7 and 15). After excluding
the active draft exit, selected and all observe the same alternative during
drafting. Terminal ordinary steps can differ. This is a correctness smoke
matrix, not a test that distinguishes an adaptive selector from all-feedback.
There are 2 GSM8K prompts, at most 8 new tokens, and 3 repeats per case.
Five unit tests additionally check native-output and teacher-forced observer
alignment, EOS, invalid inputs, and exclusion of rejected feedback.
