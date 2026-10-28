# RTX 4060 outcome

The residual-gate model does not clear the success bar. No journal manuscript is written from these numbers. Energy is NVIDIA GPU joules on the RTX 4060 Laptop. Each task uses 24 examples. Learned gates use the median of 3 measured repeats for seed 0 and 1 repeat for the other seeds.

GSM8K exact match is 1/24 for full depth and for the residual gates, about 4%. Joules per correct answer are 485 for full depth and 541, 578, and 631 for the three residual seeds. The residual gate does not use fewer joules. Mean exit depth is about 29.6 of 30 blocks.

BoolQ is the only task with a real accuracy. Full depth is 14/24 (58%) at 13.3 J per correct answer. Residual seed 0 matches that accuracy at 15.1 J per correct answer, which is higher, not lower. The entropy gate seed 0 is 17/24 at 12.3 J per correct answer. The fixed middle exit and the entropy threshold score 0 on BoolQ.

SVAMP exact match is 0/24 for every method, including full depth.

The paired intervals do not show a joule saving for the residual gate against the entropy gate or the depth-penalty gate. The gates are not skipping the remaining blocks. A claim that this checkpoint is an energy-efficient model is not supported. The 3050 measurement has not been run.
