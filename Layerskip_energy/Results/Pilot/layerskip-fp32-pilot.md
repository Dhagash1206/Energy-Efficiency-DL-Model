# Mechanics pilot report

Model: facebook/layerskip-llama3.2-1B
Revision: 81deb0f88734409cca506bcefcfb5c6bd2667565
Precision: fp32
Correctness: correctness_passed (34/34 runs agree)

**This is a mechanics check, not evidence of research energy savings.**
LayerSkip is exit-trained. CPU energy is unavailable. Timing includes prefill,
decoding, observation and trace processing; paired differences are noisy.

| Depth | Draft length | Observation | Runs | Median request seconds | Median joules |
|---|---|---|---|---|---|
| 0 | 0 | none | 2 | 3.935161 | unavailable |
| 4 | 1 | all | 2 | 4.607576 | unavailable |
| 4 | 1 | none | 2 | 4.179763 | unavailable |
| 4 | 1 | periodic | 2 | 4.451956 | unavailable |
| 4 | 1 | selected | 2 | 4.458209 | unavailable |
| 4 | 3 | all | 2 | 4.987017 | unavailable |
| 4 | 3 | none | 2 | 4.537619 | unavailable |
| 4 | 3 | periodic | 2 | 4.632391 | unavailable |
| 4 | 3 | selected | 2 | 4.971313 | unavailable |
| 8 | 1 | all | 2 | 4.224067 | unavailable |
| 8 | 1 | none | 2 | 4.013854 | unavailable |
| 8 | 1 | periodic | 2 | 4.157890 | unavailable |
| 8 | 1 | selected | 2 | 4.435910 | unavailable |
| 8 | 3 | all | 2 | 4.925115 | unavailable |
| 8 | 3 | none | 2 | 4.435891 | unavailable |
| 8 | 3 | periodic | 2 | 4.458421 | unavailable |
| 8 | 3 | selected | 2 | 4.808241 | unavailable |

## Feedback checks

Valid one-step observations: 142.
Excluded observations: 62; these are not counted as failures.

## Next decision

No go/no-go conclusion is justified from this smoke test.
Use repeated sensor-backed measurements with an exit-trained model before
deciding whether selective observation can repay its costs.
The selected mode observes the first available alternative exit; it is a
fixed baseline, not the proposed adaptive selector.

## Limits

- Pilot only; no novel selector implemented.
- SmolLM/tiny are mechanics checks, not exit-trained substitutes.
- CPU energy unavailable; GPU counters exclude CPU controller energy.
- Full traces add overhead in every mode; not optimized serving.
- Single task and small sample do not establish generalization.
