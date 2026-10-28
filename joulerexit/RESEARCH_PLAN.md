# Research plan: a model that skips layers to save measured GPU energy

## What we are making

A small language model whose own weights decide when to stop, then **physically skip** the remaining blocks. The saved checkpoint is the energy-efficient model. Easy tokens leave early. Hard tokens continue. Energy is the NVIDIA GPU’s joules, measured with NVML, not a FLOP count.

This is not a new foundation model trained from scratch. It is a trained early-exit adaptation of models that already fit an 8 GB laptop GPU. The result is still a model: the exit decision is inside the checkpoint, and skipped blocks do not run.

## What the literature left undone

Three recent papers sit next to this idea and stop short of it.

- **TIDE** (2026), *Token-Informed Depth Execution*, says its current system still runs every layer and only chooses which layer’s output to use. It names the missing piece as future work: a true skip that physically bypasses layers, which requires handling a broken key-value cache. That skip is what saves energy. Choosing an output after all layers have run does not.
- **GREEN-CODE** (2025) does use measured energy, but the energy decision is a separate reinforcement-learning agent on top of a code model. The authors’ own future work is larger code models and multi-agent software workflows, not a general model whose weights contain the energy decision.
- **AdaPonderLM** (2026) trains token-wise exit gates, then says future work should scale up and replace the gate with a supervised difficulty signal. Those gates follow loss and FLOPs. They do not see measured joules. **LayerSkip** (2024) trains middle layers to predict tokens and uses them for self-speculation. It optimizes speed, not joules.

No paper we found trains the exit gates of a general language model on a device energy profile **and** actually skips the unused blocks.

## What we already measured

On this Lenovo’s RTX 4060 Laptop GPU, SmolLM2-135M was run at blocks 8, 15, 23, and 30. The report is `joulerexit/results/rtx4060-smollm-probe.json`.

Energy is not a flat multiple of depth. A depth-only fit has error 0.288 J. Adding prompt length and prefill-versus-decode drops the error to 0.100 J. Repeat noise is 0.002 J. The fit’s decode term is about 0.57 J per generated token, while the depth slope is only about 0.003 J per extra block. Generating a token dominates the energy. Going one block deeper does not.

So the model we train must be allowed to skip the remaining decode blocks. A rule that only says “use fewer layers by a fixed amount” does not match this GPU.

## The model

Two checkpoints, both with the same exit design.

1. **SmolLM2-135M**, with the backbone lightly updated and the gates trained. This is the fully ours artifact.
2. **LayerSkip-Llama-3.2-1B** (`facebook/layerskip-llama3.2-1B`), backbone frozen, gates trained. Its middle layers can already predict tokens, which is what an early exit needs. Its own early-exit rule is one baseline. The checkpoint is Meta’s noncommercial research license and must be credited. Qwen3-0.6B stays a third run only if the first two train cleanly.

Exits, counted from block 1:

- SmolLM2-135M (30 blocks): 8, 15, 23, 30
- LayerSkip 1B (16 blocks): 4, 8, 12, 16

One shared vocabulary head. Each early exit has its own small gate. The gate reads the current token’s hidden state, how unsure the next token is, and the frozen energy residual from the probe (the joules that depth alone does not explain). Above a probability of 0.5 the token is emitted and later blocks are not executed. The last block always emits.

Keys and values are stored only for blocks that ran. A later token that goes deeper can attend only to earlier tokens that reached that depth. Training uses that same mask, so the measured model skips the same blocks it was trained to skip. This is the cache handling TIDE left as future work.

The loss is next-token error at the chosen exit, plus a coefficient times that exit’s predicted joules. The coefficient is chosen so validation accuracy stays within 2 points of the full-depth model.

## Hardware

- Device A, training and the first profile: this machine, RTX 4060 Laptop, 8 GB, Ryzen 7 8845HS, 16 GB RAM. CUDA PyTorch 2.11 is installed.
- Device B, the transfer test: a teammate’s RTX 3050. Same probe, then the 4060-trained gates measured there, then the residual refit on the 3050 only. The paper needs both chips. The 3050 runs are copied back as a JSON report. This laptop cannot see that GPU.

## Experiments

Tasks: GSM8K for training and scoring, SVAMP with no extra training, and BoolQ as the non-math task.

Controls, same exits: full depth, a fixed middle exit, an entropy gate, a depth-penalty gate, a gate that sees raw joules instead of the residual, and a logged energy controller sitting outside the model (the GREEN-CODE style baseline).

Three training seeds on SmolLM2 for the residual gate, the depth gate, and the entropy gate. LayerSkip 1B is one seed. Three measured repeats after warmup. Report median joules per correct answer and a paired bootstrap interval.

Success, and only then do we call the checkpoint energy-efficient: on the 4060, the residual-gate model stays within 2 accuracy points of the depth-penalty model and its joules-per-correct interval is lower than the entropy gate, the depth-penalty gate, and the outside controller, on GSM8K and BoolQ. SVAMP must not collapse. The same comparison on the 3050 is the second-device result. If the 3050 profile must be refit before the savings return, that refit is part of the paper, not a failure to hide.

If every gate exits at the same depth, or accuracy falls apart, we do not claim an energy-efficient model.

## Order of work

1. Done: residual math, exit gates, and the 4060 probe. The probe said continue.
2. Train SmolLM2-135M with real layer skipping and the frozen 4060 residual.
3. Repeat the successful settings on LayerSkip 1B with the backbone frozen.
4. Score GSM8K, SVAMP, and BoolQ on the 4060 with NVML.
5. Teammate runs the same probe and the held-out measurement on the RTX 3050.
6. Write the paper only if the success bar holds. Target an applied journal such as Sustainable Computing: Informatics and Systems. This is not a TNNLS or JMLR architecture paper.

## What we will not claim

We will not claim a new foundation model, savings on 7B-and-larger models, or whole-laptop energy. The counter is the NVIDIA GPU only. Training the gates spends energy once. The claim is about energy while answering.
