# Update README with latest results

> go through the codebase and update readme
> ran the below cmd and output stored in latest.json,latest.md in results folder
> python -m hardware_aware_selective_feedback.run `
>   --device cuda `
>   --objective energy `
>   --calibration 4 `
>   --test 8 `
>   --tokens 64 `
>   --repeats 3 `
>   --observer-pairs 6

<details><summary>3 previous messages</summary>

> I’ll review the code and saved results, then update the README to reflect the implementation and your CUDA energy run.
>
> The root README still describes the original Qwen compression plan. The current implementation runs LayerSkip decoding with measured energy feedback.
>
> Your saved run reports 381/381 greedy matches. Adaptive decoding used 35.3% less GPU energy per token and 42.3% less time per token than ordinary decoding in this sample. I’ll document the setup, policy comparisons, and limits, including that optional optimization flags were disabled.
>
> I found a reproducibility gap: the report records PyTorch 2.9.0 and Transformers 5.18.0, while the dependency file pins 2.6.0 and 4.50.0. Also, the adaptive policy bought no observations during held-out requests. Both details matter when interpreting and reproducing the results, so I’m making them explicit in the README.
</details>

Updated [README.md](./README.md) with the current architecture, your exact command, results table, setup, and testing instructions.

Documented the dependency-version mismatch and measurement limitations. Verified report figures and all README links. Only README.md was edited; the benchmark wasn’t rerun.

> so basically this was a research project given to us by proff (this is my first research project) so this was built on top of 2 research paper i guess
> create a md file explaining the pieline and the novelty we added and the improvements from from the recent run(latest.json,latest.md) and things that i need to know to explain the proff about the project

<details><summary>9 previous messages</summary>

> I’ll write a professor-facing guide explaining the pipeline, what your team added, what the latest results show, and likely questions you’ll need to answer. I’ll also check which papers the repository actually references so the guide separates established methods from your project’s contributions.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `LayerSkip Enabling Early Exit Inference and Self Speculative Decoding arxiv ...`
>
> </details>
>
> Do you remember the titles or links of the two papers your professor assigned? The repository points to LayerSkip and mentions CALM, but it doesn’t confirm that these were the assigned pair.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `https://arxiv.org/abs/2504.05598`
>
> </details>
>
> The results support an encouraging efficiency finding, but they don’t yet prove the proposed selective-feedback contribution: adaptive decoding used no paid observers and chose depth 12 for 47 of 48 test requests. A fixed depth-12 baseline is therefore an important next comparison.
>
> I also found prior work called DEL that already adapts exit depth and draft length. The guide will explain this overlap and frame your contribution around measured energy and the cost of feedback, without claiming those ideas are proven novel.
>
> **Do you remember the titles or links of the two papers your professor assigned? The repository points to LayerSkip and mentions CALM, but it doesn’t confirm that these were the assigned pair.**
>
> From Bits to Chips: An LLM-based Hardware-Aware Quantization Agent for Streamlined Deployment of LLMs
> this and layerskip not sure u verify it once
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `"From Bits to Chips" "Hardware-Aware Quantization"`
>
> Searched the web for `https://arxiv.org/abs/2601.03484`
>
> </details>
>
> I verified that title: [From Bits to Chips](https://arxiv.org/abs/2601.03484) introduces HAQA, an LLM-based agent for hardware-aware quantization and deployment. [LayerSkip](https://arxiv.org/abs/2404.16710) is the direct decoding foundation in your code.
>
> I’ll use those as the two papers you identified, while explaining the implementation boundary: your current experiment applies hardware-measured feedback to decoding choices; it does not reproduce HAQA’s quantization agent, and the latest run used FP32.
>
> The guide is written, including a one-minute explanation, pipeline diagram, contribution assessment, results calculations, professor Q&A, and a five-minute presentation outline. I’m checking the figures and code links now, especially the distinction between what was implemented and what the latest run actually tested.
</details>

Created [PROJECT_EXPLANATION.md](./PROJECT_EXPLANATION.md).

It includes:

- Verified paper references and their connection to your code.
- Beginner-friendly concepts and a pipeline diagram.
- Your contributions versus novelty still needing evidence.
- Latest results, calculations, and limitations.
- Professor Q&A and a five-minute presentation outline.
- Prioritized next experiments.

All results figures and local links were checked. Start with section 1 for your short explanation and section 9 for questions to prepare.

> currently our work is just hyperparameter tuning type?

**Partly, yes—the current core is online, hardware-aware hyperparameter selection.** Your controller chooses exit depth and draft length based on measured energy. Those are inference hyperparameters; it does not change the model’s weights.

But the project includes three distinct pieces:

| Piece | What it is |
|---|---|
| Choosing exit depth and draft length | Adaptive inference configuration |
| Deciding whether extra observations are worth their energy cost | Selective-feedback research hypothesis |
| Measuring energy, verifying outputs, and comparing policies | Experimental infrastructure and evaluation |

**Your latest results mainly demonstrate that a good speculative configuration saves energy compared with ordinary decoding.** They do not yet establish that your adaptive controller is better than a well-chosen fixed configuration: it selected depth 12 / length 3 for 47 of 48 requests, and that fixed baseline wasn’t independently evaluated across the full test workload.

So a fair description to your professor is:

> “Currently, we have implemented energy-aware online configuration selection around LayerSkip. Our proposed research contribution is cost-aware selective feedback, but we still need controlled experiments to demonstrate its additional benefit.”

The two most useful next comparisons are:

1. **Adaptive versus fixed depth 12 / length 3:** does adapting actually help?
2. **The same adaptive controller with versus without observer feedback:** does the additional information justify its cost?

Hyperparameter selection can itself be research. The key is demonstrating a meaningful new selection method or finding—not simply finding settings that outperform the ordinary baseline.

> is it fixed predefined or it is dynamic and hardware aware?

**It uses predefined choices, but selects among them dynamically using measured hardware costs.**

| Aspect | Current behavior |
|---|---|
| Available choices | Predefined: depths **4, 8, 12**, draft lengths **1, 3**, plus ordinary decoding |
| Selection | Dynamic: the adaptive controller chooses a configuration **before each request** |
| Learning | Updates its cost statistics after executing requests |
| Hardware awareness | Uses energy measured on the running GPU to score configurations |
| Within a request | Depth and draft length stay fixed in your latest run; optional `--adaptive-drafts` was disabled |

So the accurate description is **“dynamic, hardware-measured selection from a fixed configuration space.”**

It doesn’t automatically discover arbitrary depths or design GPU-specific kernels. Its hardware awareness comes from **measuring actual cost**, rather than choosing settings from a hard-coded “RTX 3050 → use depth 12” rule.

In your latest run, it chose depth 12 / length 3 for **47 of 48 requests**. That means a dynamic controller settled mostly on one choice—not that the choice was predefined. However, you still need experiments on other hardware to demonstrate that its decisions adapt usefully across devices.

> how to make the controller more intelligent and better

<details><summary>2 previous messages</summary>

> I’ll focus on changes that could improve decisions and strengthen your research claim, rather than just adding complexity to the controller.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `site.arxiv.org contextual bandit LinUCB practical contextual bandit approach personalized news article recommendation ...`
>
> </details>
</details>

**Make it better at predicting which configuration suits the current request, learning when conditions change, and deciding whether more information is worth its cost.** Adding a neural network or an LLM agent is not necessary yet.

For your project, I would follow this order.

**1. First establish whether a smarter controller is needed**

Run **fixed depth 12 / draft length 3** against your current adaptive controller under identical conditions.

Your controller chose that configuration for 47 of 48 requests. If fixed depth 12 performs equally well, the present workload doesn’t demonstrate a benefit from adaptation. You need workloads where different configurations actually win—not a controller forced to switch unnecessarily.

**2. Make decisions depend on the request**

Currently, the default controller mainly learns average costs across requests. Improve it from:

> “Depth 12 was cheapest overall.”

to:

> “For this prompt length, output budget, and current hardware condition, depth 8 is likely cheaper.”

Start with inexpensive features:

| Information | Why it could help |
|---|---|
| Prompt token count | Prefill cost changes with input length |
| Requested output budget | Short outputs leave less time to recover setup costs |
| Recent acceptance rates | Indicate whether speculation is paying off |
| Recent measured J/token and latency | Capture changes in actual execution cost |
| GPU clock and temperature, where available | Help identify throttling or changing conditions |

Use only information available **before the decision**. For example, a new request’s eventual acceptance rate cannot be an input to its initial choice.

Your existing `--context-selector` is a useful first experiment: it separates costs by dataset and short/long prompts. For deployment, avoid depending on benchmark dataset labels unless equivalent task information is actually available.

**3. Add deliberate exploration**

Your current adaptive score penalizes uncertainty. This can make poorly sampled configurations remain unattractive.

A **contextual bandit** would learn the relationship between request features and action costs while occasionally trying uncertain alternatives. Start with a small linear model, such as a cost-minimizing adaptation of LinUCB, rather than deep reinforcement learning. Contextual bandits are an established approach; using one is not itself a novelty claim. [Original LinUCB paper](https://arxiv.org/abs/1003.0146)

Conceptually:

```text
Observe request features
        ↓
Predict energy cost and uncertainty for each action
        ↓
Choose a promising action, with bounded exploration
        ↓
Execute and verify output
        ↓
Update using actual measured cost
```

Exploration consumes energy, so count that cost in the evaluation.

**4. Adapt when the workload or hardware changes**

A lifetime average can become stale. Compare it against a rolling window or an exponentially weighted average, where recent measurements receive more weight.

For example, if GPU throttling changes the relative costs, the controller should reconsider its previous choice. Add a margin before switching to avoid reacting to tiny measurement fluctuations.

This is a proposed improvement—not evidence that throttling caused your current results.

**5. Make selective feedback the focused research question**

This is the most interesting part of your proposed contribution:

> **When does paying for an observation improve future decisions enough to recover its cost?**

A useful design target is:

```text
Buy feedback when:

estimated future energy saved by better decisions
    >
observation energy + decision overhead + uncertainty margin
```

That expression is a research objective, not an already available estimator. You would need to learn and validate whether observations actually change decisions beneficially.

In particular:

- Collect feedback when competing actions have similar predicted costs and uncertainty could change the choice.
- Avoid buying information when one action clearly dominates.
- Consider how many future requests could benefit.
- Keep committed-prefix validation.
- Never treat an observer’s token agreement as a measurement of an unexecuted action’s energy.

Compare **the same controller** with feedback disabled, selective feedback, and always-on feedback. Your fixed `never_observe` policy is not enough to isolate this effect.

**6. Add within-request adaptation afterward**

Your optional `--adaptive-drafts` already offers a first step: shorten drafts when acceptance falls and periodically probe for recovery.

Test that separately before changing exit depths mid-request, which complicates cache handling. Also, dynamic exit depth and speculation length already appear in DEL, so this feature alone would not establish novelty. [DEL paper](https://arxiv.org/abs/2504.05598)

**My recommendation:** build a **lightweight contextual energy controller with cost-aware feedback acquisition**, in stages:

1. Strong fixed baseline.
2. Existing context selector.
3. Contextual bandit without observers.
4. Same bandit with selective observers.
5. Optional adaptive draft lengths.

Judge success by **lower total measured cost—including learning and observation overhead—with preserved output correctness**, not by how often the controller changes settings. A good controller may correctly stay with one configuration.

> this was done on top of layer skip

**Yes—DEL builds on early-exit self-speculative decoding, including LayerSkip.** That makes it particularly relevant prior work for your project. :codex-annotation{index="1"}

The relationship is:

```text
LayerSkip: draft using early layers, verify using remaining layers
    │
    ├── DEL: dynamically choose exit depth and draft length
    │
    └── Your project: choose configurations using measured energy
                      and investigate whether extra feedback is worth its cost
```

Building on LayerSkip is completely valid research. The question is **what your extension adds beyond both LayerSkip and existing extensions such as DEL**.

Your potential distinction is **energy-aware decisions plus cost-aware selective feedback**, rather than dynamic depth selection alone. But it remains a proposed contribution: the latest run shows lower energy than ordinary decoding, not yet an advantage over DEL or a strong fixed depth-12 baseline.

A useful explanation to your professor:

> “Like DEL, we build on LayerSkip. Our intended focus is measured GPU energy and the cost of acquiring feedback. We need controlled comparisons to establish whether those additions improve on existing configuration-selection methods.”

> can we add del work on our work

<details><summary>2 previous messages</summary>

> I’ll check DEL’s implementation and license against your engine. The best starting point is a separate DEL comparison policy, so you can measure whether your energy-aware controller adds value beyond DEL.
>
> <details><summary>Ran a command, searched the web</summary>
>
> Searched the web for `https://github.com/hoenza/DEL`
>
> - Ran `git status --short; Get-Content hardware_aware_selective_feedback/selector.py -TotalCount 40; Get-Content hardware_aware_selective_feedback/vendor/LICENSE.LayerSkip -TotalCount 20`
>
> </details>
</details>

**Yes. DEL is a relevant extension to integrate because it already builds on LayerSkip.** Its official implementation dynamically selects exit depth and speculation length using acceptance feedback. [DEL repository](https://github.com/hoenza/DEL)

I recommend keeping **two separate policies**:

| Policy | Purpose |
|---|---|
| **DEL baseline** | Reproduce DEL’s original decision rules as faithfully as possible |
| **Your energy-aware extension** | Investigate whether measured energy and selective feedback improve those decisions |

The resulting research structure would be:

```text
LayerSkip: draft and verify
        ↓
DEL: dynamically choose depth and draft length
        ↓
Your extension: account for measured energy
               and the cost of collecting feedback
```

Integration requires more than replacing `selector.py`. Your current selector chooses settings **per request**; DEL adapts during generation. We would need to extend the generation loop, maintain acceptance statistics, and ensure changing depth preserves correct KV-cache state.

A sensible implementation sequence is:

1. Add DEL as a separate policy, with attribution and verified code-reuse terms.
2. Test its token outputs against greedy decoding.
3. Benchmark ordinary decoding, fixed depth 12, DEL, and your current controller under identical conditions.
4. Add energy-aware scoring and selective feedback as separate changes, measuring each contribution.

**Adding DEL would strengthen your baseline and foundation; it would not itself be your novelty.** The research question becomes:

> “Can hardware-measured energy and cost-aware feedback improve the efficiency of DEL-based self-speculative decoding?”

That is a clearer, testable direction. It may improve results, but we should establish that experimentally rather than assume it.

> any gaps in del that we can contribute to

<details><summary>2 previous messages</summary>

> I’ll check DEL’s method and evaluation for concrete gaps, then separate promising research questions from things the paper already addresses.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `https://arxiv.org/html/2504.05598v2`
>
> Searched the web for `'overhead' in https://arxiv.org/html/2504.05598v2`
>
> </details>
</details>

**Yes. The strongest direction for your project is testing whether DEL’s decisions remain efficient when evaluated by actual GPU energy and feedback cost.** These are research opportunities—not yet proven weaknesses or globally novel ideas.

One correction to my earlier guidance: **DEL already filters feedback to valid contexts and gives recent observations more weight.** Those should not be presented as additions beyond DEL. [DEL, §4.2](https://arxiv.org/html/2504.05598v2#S4.SS2)

**1. Replace the compute proxy with a measured energy model — strongest fit**

DEL selects configurations using *tokens per loaded layer*, a proxy based on an assumed relationship between layer loading and execution cost. Its reported objectives are throughput and speedup, rather than measured joules. [DEL, §4.1 and §5.1](https://arxiv.org/html/2504.05598v2#S4.SS1)

Your research question:

> Does the configuration that maximizes tokens per layer also minimize GPU energy per token?

Proposed extension:

```text
Choose the action with the lowest:

predicted energy for drafting, verification, and controller work
────────────────────────────────────────────────────────────────
              expected committed output tokens
```

Start by measuring whether DEL’s configuration ranking differs from an energy-based ranking. **If the rankings rarely differ, changing the objective may offer little benefit.** That initial experiment should come before building a complex predictor.

**2. Selectively collect intermediate-layer feedback — strongest algorithmic opportunity**

DEL projects cached hidden states from all layers through the LM head to obtain shadow predictions. It already measures runtime and memory overhead and reports it as small in its experiments. Therefore, “DEL ignores overhead” would be inaccurate. [DEL, §4.2 and §5.3](https://arxiv.org/html/2504.05598v2#S4.SS2)

Your research question:

> Can we observe fewer layers or observe less frequently without losing useful configuration decisions?

A possible policy would:

- Observe competing exit depths when their predicted costs are close.
- Reduce observations when one configuration consistently wins.
- Periodically refresh neglected alternatives.
- Count observation cost and the cost of worse decisions caused by stale information.

**Important:** your current observer overhead measurements are not measurements of DEL’s implementation. You must first reproduce its efficient observation path before claiming an improvement.

**3. Test adaptation under consumer-GPU operating conditions — useful systems contribution**

DEL’s reported experiments use A100/H100 GPUs. Your RTX 3050 offers a different evaluation setting. [DEL, Appendix B.2](https://arxiv.org/html/2504.05598v2#A2.SS2)

Investigate whether configuration rankings change with:

- Sustained execution and temperature changes.
- Supported clock or power settings.
- Prompt length and output budget.

The contribution would be **a measured explanation and a controller that responds usefully**, not merely running DEL on a laptop. Keep conditions controlled and record hardware state.

**4. Minimize energy subject to a latency requirement — a further extension**

A practical proposed objective is:

> Choose the lowest-energy configuration that meets a user-defined latency budget.

Evaluate an energy–latency tradeoff curve rather than one headline saving. This may be useful, but it broadens the project; I would attempt it after the first two experiments.

For your first research project, I recommend this focused question:

> **Can an energy-aware DEL extension with selective intermediate-layer observations reduce total GPU energy while preserving greedy output and acceptable latency?**

Use a small experiment matrix to isolate the contributions:

| Policy | Configuration objective | Feedback |
|---|---|---|
| Original DEL | Original TPL rule | Original observation mechanism |
| Energy-aware variant | Predicted measured energy | Same observation mechanism |
| Selective-feedback variant | Original TPL rule | Selective observations |
| Combined variant | Predicted measured energy | Selective observations |

Also include ordinary decoding and a calibration-selected fixed baseline.

Your current **35.3% saving versus ordinary decoding is motivation**, not evidence that these extensions beat DEL. The next meaningful milestone is a faithful DEL baseline and a measurement showing **where its decisions leave recoverable energy savings**.

> can we do del but for consumer hardware

**Yes—adapting DEL for consumer hardware is a realistic direction for your project.** Your RTX 3050 and existing energy-measurement infrastructure give you a starting point.

The distinction is:

- **Running DEL on an RTX 3050:** a reproduction and hardware evaluation.
- **Identifying consumer-hardware limitations and modifying DEL to address them:** a potential research contribution.

A focused research question would be:

> **How can DEL be adapted to minimize GPU energy on resource-constrained consumer hardware while preserving greedy output?**

I suggest three stages:

| Stage | Work | What it establishes |
|---|---|---|
| **1. Reproduce** | Run original DEL alongside ordinary decoding and a tuned fixed LayerSkip baseline | Whether DEL helps on your GPU |
| **2. Diagnose** | Measure J/token, latency, peak VRAM, and observation overhead across prompt/output lengths | Where the original approach has room for improvement |
| **3. Adapt** | Address the measured bottleneck through energy-aware selection, selective observations, or reduced temporary memory | Whether your changes provide additional benefit |

**Choose the modification after measuring the bottleneck.** For example, if DEL’s observations are inexpensive on your GPU, selective observation may offer little benefit. If its preferred configurations consume more energy than alternatives, energy-aware selection becomes a stronger direction.

For a manageable first version:

- Use the existing LayerSkip 1B checkpoint.
- Start with greedy decoding and one validated precision.
- Preserve original DEL as a separate baseline.
- Include controller and observation costs in measurements.
- Evaluate more prompts and repeat complete runs.
- Test another consumer GPU if available; otherwise describe the findings specifically as **RTX 3050 results**.

A suitable working title is:

**“Adapting Dynamic Self-Speculative Decoding for Energy-Efficient Inference on Consumer GPUs.”**

This builds naturally on your existing work. The contribution would be the **measured adaptation and evidence explaining when it helps**, rather than the hardware change alone.

> so ours is like del but worse cause its choosing from predefined stuff instead of dynamically doing it?

**Your current controller is less fine-grained than DEL, but “predefined choices” isn’t the reason.** Both methods choose among discrete options—exit layers and draft lengths.

The main differences are:

| Aspect | Your current controller | DEL |
|---|---|---|
| When it chooses | Before each request | During generation, at speculative-round granularity |
| Available choices | Depths 4/8/12 and lengths 1/3, plus ordinary | Broader layer and draft-length selection |
| Decision basis | Measured energy or latency from executed requests | Acceptance estimates and tokens-per-layer efficiency |
| Within-request response | Configuration stays fixed in your latest run | Responds as the generated context changes |
| Extra feedback | Restricted observer choices with a cost heuristic | Layer-wise shadow predictions |

So **your implementation is simpler and less responsive within a request, while its objective is more directly tied to measured hardware energy.** That difference does not automatically make it better—or worse—on energy.

Also, selecting depth 12 for 47 requests is not itself a failure. If that configuration is consistently cheapest, staying with it is correct.

The honest assessment is:

> “We currently have a coarse, request-level energy-aware controller. DEL provides more sophisticated within-generation adaptation. We have not yet compared them, so we cannot claim that our controller performs better.”

Your next step should be **implementing DEL as a baseline and testing whether your measured-energy approach adds value**, rather than making the controller switch more often just to appear dynamic.

> create new branch vinanth_patch4 and push the current version

<details><summary>4 previous messages</summary>

> I’ll create `vinanth_patch4`, commit the current changes, and push the branch to the repository remote.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `git status --short; git branch --show-current; git remote -v; git branch --list vinanth_patch4; git diff --stat; git ls-files --others --exclude-standard`
> - Ran `git diff -- .gitignore hardware_aware_selective_feedback/vendor/self_speculation/llama_model_utils.py; git diff --check; git ls-remote --heads origin vinanth_patch4`
>
> </details>
>
> Your `.gitignore` appears intended to exclude the explanation guide, but its filename is misspelled. I’ll correct that entry and keep the guide local. The push also needs network access beyond the sandbox, so I’ll request that access.
>
> <details><summary>Ran 7 commands</summary>
>
> - Ran `$ignore = Get-Content .gitignore -Raw; $ignore.Replace('PROJECT_EXPLAINATION.md','PROJECT_EXPLANATION.md') | Set-Content .gitignore -Encoding ascii; git switch -c vinanth_patch4`
> - Ran `git switch -c vinanth_patch4`
> - Ran `git add -- .gitignore README.md hardware_aware_selective_feedback/results/latest.json hardware_aware_selective_feedback/results/latest.md hardware_aware_selective_feedback/results/smoke/rtx3050-fp16-smoke.json hardware_aware_selective_feedback/results/smoke/rtx3050-fp16-smoke.md hardware_aware_selective_feedback/vendor/self_speculation/llama_model_utils.py`
> - Ran `git diff --cached --check; git diff --cached --stat; git status --short`
> - Ran `git commit -m "Save CUDA energy results, runtime compatibility, and project documentation"`
> - Ran `git push -u origin vinanth_patch4`
> - Ran `git status --short; git log -1 --format='%h %s'; git rev-parse --abbrev-ref --symbolic-full-name '@{u}'`
>
> </details>
</details>

Created and pushed [vinanth_patch4](https://github.com/Dhagash1206/Energy-Efficiency-DL-Model/tree/vinanth_patch4).

- Commit: `aff6b54`
- Working tree is clean.
- Corrected the `.gitignore` typo; `PROJECT_EXPLANATION.md` remains local and wasn’t pushed.

> do this

<details><summary>15 previous messages</summary>

> I’ll add DEL as a separate baseline, verify its decoding correctness, and compare it with your energy-aware controller under matched settings. I’ll first check the upstream algorithm and code terms so any deviations are documented accurately. :codex-annotation{index="1"}
>
> <details><summary>Ran 3 commands, called 4 tools, searched the web</summary>
>
> - Ran `rg --files -g AGENTS.md; Get-Content hardware_aware_selective_feedback/engine.py; Get-Content hardware_aware_selective_feedback/sweep.py`
>
> Searched the web for `https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/autoregressive.py`
>
> Searched the web for `https://api.github.com/repos/hoenza/DEL/contents/self_speculation`
>
> - Ran `Get-Content hardware_aware_selective_feedback/tests/test_engine.py -TotalCount 100; Get-Content hardware_aware_selective_feedback/run.py | Select-Object -Skip 230 -First 125; Get-Command python | Select-Object Source; Get-ChildItem -Force -Name; Get-Content hardware_aware_selective_feedback/vendor/self_speculation/llama_model_utils.py | Select-Object -Skip 240 -First 180`
>
> - Ran `.\.venv\Scripts\python.exe -c "import torch, transformers; print(torch.__version__, transformers.__version__); print('CUDA',torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"; Get-Content .gitignore; Get-Content hardware_aware_selective_feedback/tests/test_run.py -TotalCount 120`
>
> MCP tool call
>
> codex_apps.github.fetch
>
> ```json
> {
>   "url": "https://github.com/hoenza/DEL/tree/main/self_speculation"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "[{\"name\":\"DEL.py\",\"path\":\"self_speculation/DEL.py\",\"sha\":\"8c3a3c02968e963ad3ca91fa72ade726b6cc55b5\",\"size\":10204,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/8c3a3c02968e963ad3ca91fa72ade726b6cc55b5\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DEL.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/8c3a3c02968e963ad3ca91fa72ade726b6cc55b5\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py\"}},{\"name\":\"DELE_speculation_generator.py\",\"path\":\"self_speculation/DELE_speculation_generator.py\",\"sha\":\"18b558486a1847477bafe980ab5791dcfcd6f4a3\",\"size\":10829,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DELE_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DELE_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/18b558486a1847477bafe980ab5791dcfcd6f4a3\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DELE_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DELE_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/18b558486a1847477bafe980ab5791dcfcd6f4a3\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DELE_speculation_generator.py\"}},{\"name\":\"DEL_speculation_generator.py\",\"path\":\"self_speculation/DEL_speculation_generator.py\",\"sha\":\"3e0e284bed3be71b456ab5d293f595975a8efa01\",\"size\":11545,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/3e0e284bed3be71b456ab5d293f595975a8efa01\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DEL_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/3e0e284bed3be71b456ab5d293f595975a8efa01\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py\"}},{\"name\":\"DV_speculation_generator.py\",\"path\":\"self_speculation/DV_speculation_generator.py\",\"sha\":\"1a88602e935fad897aa733dd606aba2530c07a36\",\"size\":11138,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DV_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DV_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1a88602e935fad897aa733dd606aba2530c07a36\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DV_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DV_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1a88602e935fad897aa733dd606aba2530c07a36\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DV_speculation_generator.py\"}},{\"name\":\"FSM_speculation_generator.py\",\"path\":\"self_speculation/FSM_speculation_generator.py\",\"sha\":\"7fd6edaf456bf1c61d0a52cb9de0f10dead415b7\",\"size\":10415,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/FSM_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/FSM_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/7fd6edaf456bf1c61d0a52cb9de0f10dead415b7\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/FSM_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/FSM_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/7fd6edaf456bf1c61d0a52cb9de0f10dead415b7\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/FSM_speculation_generator.py\"}},{\"name\":\"__init__.py\",\"path\":\"self_speculation/__init__.py\",\"sha\":\"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\",\"size\":0,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/__init__.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/__init__.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/__init__.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/__init__.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/__init__.py\"}},{\"name\":\"autoregressive_generator.py\",\"path\":\"self_speculation/autoregressive_generator.py\",\"sha\":\"bfe1817e3562beaa08b08657406e1c85dd5993f0\",\"size\":3014,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/autoregressive_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/autoregressive_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/bfe1817e3562beaa08b08657406e1c85dd5993f0\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/autoregressive_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/autoregressive_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/bfe1817e3562beaa08b08657406e1c85dd5993f0\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/autoregressive_generator.py\"}},{\"name\":\"generator_base.py\",\"path\":\"self_speculation/generator_base.py\",\"sha\":\"af107b8b65601eaa21e5ced5fa9cf2e1ecff8e48\",\"size\":4855,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/generator_base.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/generator_base.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/af107b8b65601eaa21e5ced5fa9cf2e1ecff8e48\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/generator_base.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/generator_base.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/af107b8b65601eaa21e5ced5fa9cf2e1ecff8e48\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/generator_base.py\"}},{\"name\":\"llama_model_utils.py\",\"path\":\"self_speculation/llama_model_utils.py\",\"sha\":\"fb0fb2d40861940f220496b5b79375a45d4fca34\",\"size\":22380,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/llama_model_utils.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/llama_model_utils.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/fb0fb2d40861940f220496b5b79375a45d4fca34\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/llama_model_utils.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/llama_model_utils.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/fb0fb2d40861940f220496b5b79375a45d4fca34\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/llama_model_utils.py\"}},{\"name\":\"self_speculation_generator.py\",\"path\":\"self_speculation/self_speculation_generator.py\",\"sha\":\"1e76d564d1e4bfe874e10e32e3a1921a8fde8fe3\",\"size\":10221,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/self_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/self_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1e76d564d1e4bfe874e10e32e3a1921a8fde8fe3\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/self_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/self_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1e76d564d1e4bfe874e10e32e3a1921a8fde8fe3\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/self_speculation_generator.py\"}},{\"name\":\"speculative_streamer.py\",\"path\":\"self_speculation/speculative_streamer.py\",\"sha\":\"9b28ba293494384811ad8e0e5a9aa3939b5817b4\",\"size\":3292,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/speculative_streamer.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/speculative_streamer.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/9b28ba293494384811ad8e0e5a9aa3939b5817b4\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/speculative_streamer.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/speculative_streamer.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/9b28ba293494384811ad8e0e5a9aa3939b5817b4\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/speculative_streamer.py\"}}]",
>   "title": "self_speculation",
>   "modified_date": null,
>   "url": "https://github.com/hoenza/DEL/tree/main/self_speculation",
>   "display_url": "https://github.com/hoenza/DEL/tree/main/self_speculation",
>   "display_title": "self_speculation",
>   "structuredContent": {
>     "title": "self_speculation",
>     "content": "[{\"name\":\"DEL.py\",\"path\":\"self_speculation/DEL.py\",\"sha\":\"8c3a3c02968e963ad3ca91fa72ade726b6cc55b5\",\"size\":10204,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/8c3a3c02968e963ad3ca91fa72ade726b6cc55b5\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DEL.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/8c3a3c02968e963ad3ca91fa72ade726b6cc55b5\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py\"}},{\"name\":\"DELE_speculation_generator.py\",\"path\":\"self_speculation/DELE_speculation_generator.py\",\"sha\":\"18b558486a1847477bafe980ab5791dcfcd6f4a3\",\"size\":10829,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DELE_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DELE_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/18b558486a1847477bafe980ab5791dcfcd6f4a3\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DELE_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DELE_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/18b558486a1847477bafe980ab5791dcfcd6f4a3\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DELE_speculation_generator.py\"}},{\"name\":\"DEL_speculation_generator.py\",\"path\":\"self_speculation/DEL_speculation_generator.py\",\"sha\":\"3e0e284bed3be71b456ab5d293f595975a8efa01\",\"size\":11545,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/3e0e284bed3be71b456ab5d293f595975a8efa01\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DEL_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DEL_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/3e0e284bed3be71b456ab5d293f595975a8efa01\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py\"}},{\"name\":\"DV_speculation_generator.py\",\"path\":\"self_speculation/DV_speculation_generator.py\",\"sha\":\"1a88602e935fad897aa733dd606aba2530c07a36\",\"size\":11138,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DV_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DV_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1a88602e935fad897aa733dd606aba2530c07a36\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/DV_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/DV_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1a88602e935fad897aa733dd606aba2530c07a36\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/DV_speculation_generator.py\"}},{\"name\":\"FSM_speculation_generator.py\",\"path\":\"self_speculation/FSM_speculation_generator.py\",\"sha\":\"7fd6edaf456bf1c61d0a52cb9de0f10dead415b7\",\"size\":10415,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/FSM_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/FSM_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/7fd6edaf456bf1c61d0a52cb9de0f10dead415b7\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/FSM_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/FSM_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/7fd6edaf456bf1c61d0a52cb9de0f10dead415b7\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/FSM_speculation_generator.py\"}},{\"name\":\"__init__.py\",\"path\":\"self_speculation/__init__.py\",\"sha\":\"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\",\"size\":0,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/__init__.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/__init__.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/__init__.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/__init__.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/e69de29bb2d1d6434b8b29ae775ad8c2e48c5391\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/__init__.py\"}},{\"name\":\"autoregressive_generator.py\",\"path\":\"self_speculation/autoregressive_generator.py\",\"sha\":\"bfe1817e3562beaa08b08657406e1c85dd5993f0\",\"size\":3014,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/autoregressive_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/autoregressive_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/bfe1817e3562beaa08b08657406e1c85dd5993f0\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/autoregressive_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/autoregressive_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/bfe1817e3562beaa08b08657406e1c85dd5993f0\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/autoregressive_generator.py\"}},{\"name\":\"generator_base.py\",\"path\":\"self_speculation/generator_base.py\",\"sha\":\"af107b8b65601eaa21e5ced5fa9cf2e1ecff8e48\",\"size\":4855,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/generator_base.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/generator_base.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/af107b8b65601eaa21e5ced5fa9cf2e1ecff8e48\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/generator_base.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/generator_base.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/af107b8b65601eaa21e5ced5fa9cf2e1ecff8e48\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/generator_base.py\"}},{\"name\":\"llama_model_utils.py\",\"path\":\"self_speculation/llama_model_utils.py\",\"sha\":\"fb0fb2d40861940f220496b5b79375a45d4fca34\",\"size\":22380,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/llama_model_utils.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/llama_model_utils.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/fb0fb2d40861940f220496b5b79375a45d4fca34\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/llama_model_utils.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/llama_model_utils.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/fb0fb2d40861940f220496b5b79375a45d4fca34\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/llama_model_utils.py\"}},{\"name\":\"self_speculation_generator.py\",\"path\":\"self_speculation/self_speculation_generator.py\",\"sha\":\"1e76d564d1e4bfe874e10e32e3a1921a8fde8fe3\",\"size\":10221,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/self_speculation_generator.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/self_speculation_generator.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1e76d564d1e4bfe874e10e32e3a1921a8fde8fe3\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/self_speculation_generator.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/self_speculation_generator.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/1e76d564d1e4bfe874e10e32e3a1921a8fde8fe3\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/self_speculation_generator.py\"}},{\"name\":\"speculative_streamer.py\",\"path\":\"self_speculation/speculative_streamer.py\",\"sha\":\"9b28ba293494384811ad8e0e5a9aa3939b5817b4\",\"size\":3292,\"url\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/speculative_streamer.py?ref=main\",\"html_url\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/speculative_streamer.py\",\"git_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/9b28ba293494384811ad8e0e5a9aa3939b5817b4\",\"download_url\":\"https://raw.githubusercontent.com/hoenza/DEL/main/self_speculation/speculative_streamer.py\",\"type\":\"file\",\"_links\":{\"self\":\"https://api.github.com/repos/hoenza/DEL/contents/self_speculation/speculative_streamer.py?ref=main\",\"git\":\"https://api.github.com/repos/hoenza/DEL/git/blobs/9b28ba293494384811ad8e0e5a9aa3939b5817b4\",\"html\":\"https://github.com/hoenza/DEL/blob/main/self_speculation/speculative_streamer.py\"}}]",
>     "display_url": "https://github.com/hoenza/DEL/tree/main/self_speculation"
>   }
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch
>
> ```json
> {
>   "url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "import torch\nimport itertools\nfrom typing import List, Optional, Tuple\n\nfrom self_speculation.llama_model_utils import (\n    decode_next_token,\n)\n\n\nclass DEL:\n    def __init__(self, model, gamma_max, eligible_exit_layers=None, omega=1.0, gamma_min=0):\n        self.model = model\n        self.token_match_counts = torch.zeros((len(model.model.layers), 2))\n        self.confidence_sums = torch.zeros((len(model.model.layers), 2))\n        self.average_confidence = torch.zeros((len(model.model.layers), 2))\n        self.is_prefill_stage = True\n        self.omega = omega\n        if eligible_exit_layers is None:\n            self.eligible_exit_layers = list(range(0, len(self.model.model.layers)-1))\n        else:\n            self.eligible_exit_layers = eligible_exit_layers\n        self.gamma_max = gamma_max\n        self.gamma_min = gamma_min\n        self.cached_hidden_states = [[] for i in range(len(self.model.model.layers))]\n        self.current_exit_layer = self.eligible_exit_layers[0]+1\n        self.current_gamma = 1\n\n    def clear_cached_hidden_states(self):\n        \"\"\"Reset cached hidden states for all layers (used at the start of each SD round).\"\"\"\n        self.cached_hidden_states = [[] for i in range(len(self.model.model.layers))]\n    \n    def remove_old_hidden_states(self, input_length):\n        \"\"\"\n        Trim outdated cached hidden states based on consumed input length.\n        Applies a 32-token prefill adjustment on first use.\n        \"\"\"\n        if self.is_prefill_stage:\n            input_length = max(0, input_length - 32)\n            self.is_prefill_stage = False\n        if input_length == 0:\n            return 0\n        \n        for i in self.eligible_exit_layers:\n            rmln = input_length\n            if self.cached_hidden_states[i]:\n                new_hidden_states = []\n                for j in range(len(self.cached_hidden_states[i])):\n                    if rmln == 0:\n                        new_hidden_states.append(self.cached_hidden_states[i][j])\n                    elif rmln >= self.cached_hidden_states[i][j].shape[1]:\n                        rmln -= self.cached_hidden_states[i][j].shape[1]\n                    else:\n                        new_hidden_states.append(self.cached_hidden_states[i][j][:, rmln:, :])\n                        rmln = 0\n                self.cached_hidden_states[i] = new_hidden_states\n        return input_length\n\n    def update_acceptance_and_confidence_stats(self, tokens, shadow_confidence_scores, exit_layer, context_length=0):\n        \"\"\"\n        Update acceptance counts (α_ℓ) and confidence score statistics (τ_ℓ) \n        using shadow tokens from each candidate draft layer.\n        \"\"\"\n        # All but the last row are predictions from candidate draft layers.\n        layers_tokens = tokens[:-1, :]\n        \n        # Last row is the target model’s predictions (full-depth layer).\n        correct_tokens = tokens[-1, :].unsqueeze(0).expand_as(layers_tokens)\n        \n        # Boolean table indicating where layer predictions match the target model\n        token_match_table = layers_tokens == correct_tokens\n\n        # Split context tokens if present\n        if context_length > 0:\n            context_token_match_table = token_match_table[:, :context_length]\n            token_match_table = token_match_table[:, context_length:]\n            context_shadow_confidence_scores = shadow_confidence_scores[:, :context_length]\n            shadow_confidence_scores = shadow_confidence_scores[:, context_length:]\n        \n\n        # Clip tokens after the first mismatch at the current exit layer\n        if self.current_gamma > 0:\n            current_can_layer = exit_layer - 1 - self.eligible_exit_layers[0]\n            current_can_layer_false_indices = (token_match_table[current_can_layer, :] == False).nonzero(as_tuple=True)[0]\n            if len(current_can_layer_false_indices) > 0:\n                token_match_table = token_match_table[:, :current_can_layer_false_indices[0].item()+1]\n                shadow_confidence_scores = shadow_confidence_scores[:, :current_can_layer_false_indices[0].item()+1]\n        \n        # Restore context if it was separated\n        if context_length > 0:\n            token_match_table = torch.cat((context_token_match_table, token_match_table), dim=1)\n            shadow_confidence_scores = torch.cat((context_shadow_confidence_scores, shadow_confidence_scores), dim=1)\n        \n        # Sum confidence scores over matched and mismatched tokens\n        true_shadow_confidence_scores = (shadow_confidence_scores * token_match_table.float()).sum(dim=1)\n        false_shadow_confidence_scores = (shadow_confidence_scores * (~token_match_table).float()).sum(dim=1)\n        confidence_score_sums_update = torch.stack((true_shadow_confidence_scores, false_shadow_confidence_scores), dim=1)\n        \n        # Apply exponential moving average update for confidence scores\n        self.confidence_sums[self.eligible_exit_layers, :] = (\n            self.omega * self.confidence_sums[self.eligible_exit_layers, :] + confidence_score_sums_update.to(self.confidence_sums)\n        )\n        \n        # Count number of matches/mismatches per layer\n        true_counts = token_match_table.sum(dim=1)\n        false_counts = token_match_table.shape[1]-true_counts\n        match_counts_update = torch.stack((true_counts, false_counts), dim=1)\n        \n        # Update moving average for token matches\n        self.token_match_counts[self.eligible_exit_layers, :] = (\n            self.omega * self.token_match_counts[self.eligible_exit_layers, :] + match_counts_update.to(self.token_match_counts)\n        )\n        \n        # Derive average confidence per layer over matched/mismatched tokens\n        self.average_confidence = self.confidence_sums / self.token_match_counts\n    \n    def select_optimal_exit_and_gamma(self):\n        \"\"\"\n        Compute TPL(ℓ, γ) for all candidate exit layers and speculation lengths,\n        and select the configuration that maximizes it.\n        \"\"\"\n        \n        # Estimate α_ℓ = accepted / total tokens for each candidate layer\n        probs = self.token_match_counts[self.eligible_exit_layers, 0] / torch.sum(self.token_match_counts[self.eligible_exit_layers, :], dim=1)\n        probs = torch.where(torch.isnan(probs), torch.tensor(float('-inf')), probs)\n        \n        # Compute expected number of accepted tokens per speculation length using α_ℓ\n        gamma_probs = torch.zeros((len(self.eligible_exit_layers), self.gamma_max))\n        for i in range(self.gamma_max):\n            if i == 0:\n                gamma_probs[:, i] = 1\n            else:\n                gamma_probs[:, i] = 1 + probs * gamma_probs[:, i-1]\n        \n        # Cost model: each round costs γ * draft_cost + verification_cost\n        draft_cost = torch.tensor(self.eligible_exit_layers) + 1\n        verify_cost = len(self.model.model.layers) - draft_cost\n        layers_costs = torch.zeros((len(self.eligible_exit_layers), self.gamma_max))\n        for i in range(self.gamma_max):\n            layers_costs[:, i] = verify_cost + (i+1) * draft_cost\n        \n        # Compute benefit = expected tokens / total cost = TPL\n        tpl = gamma_probs / layers_costs\n        \n        # Enforce minimum speculation length\n        if self.gamma_min > 0:\n            tpl[:, :self.gamma_min] = float('-inf')\n            \n        # Select (layer, γ) pair that maximizes TPL\n        del_layer, del_gamma = divmod(torch.argmax(tpl).item(), tpl.shape[1])\n        \n        # Convert back to global layer index\n        self.current_exit_layer = del_layer + self.eligible_exit_layers[0] + 1\n        self.current_gamma = del_gamma\n\n    def run(self, exit_layer, input_length, sample):\n        \"\"\"\n        Finalize verification of the current speculative decoding round:\n        - Compute shadow logits\n        - Derive token match and confidence stats\n        - Update optimal exit layer and speculation length (E, γ)\n        \"\"\"\n        \n        # Compute number of new tokens generated\n        output_length = self.cached_hidden_states[-1][0].shape[1] - input_length\n        \n        # Remove past tokens (prefill or already processed)\n        trim_length = self.remove_old_hidden_states(input_length)\n        context_length = input_length - trim_length\n\n        # Flatten cached hidden states from all layers\n        flattened_hiddens = list(itertools.chain.from_iterable(self.cached_hidden_states))\n        if not flattened_hiddens:\n            return None, None\n        \n        # Send all to LM-head (on correct device)\n        lm_head_device = self.model.lm_head.weight.device\n        flattened_hiddens = [h.to(lm_head_device) for h in flattened_hiddens]\n        hiddens = torch.cat(flattened_hiddens, dim=1)\n        hiddens = self.model.model.norm(hiddens)\n        \n        # Compute logits and extract segment of interest\n        logits = self.model.lm_head(hiddens)\n        return_logits = logits[:, -(input_length+output_length):, :]\n        logits = torch.cat(\n            (logits[:, 0:-(input_length + output_length), :], logits[:, -(context_length + output_length):, :]),\n            dim=1\n        )\n        \n        # Greedy decode shadow tokens and extract their confidence scores\n        tokens, shadow_confidence_scores = decode_next_token(logits=logits, sample=False)\n        shadow_confidence_scores = shadow_confidence_scores[torch.arange(shadow_confidence_scores.size(0)), tokens.squeeze(0)]\n        \n        # Only return generated tokens (exclude context) if not sampling\n        return_tokens = tokens[:, -output_length:] if not sample else None\n        \n        # Reshape to [#layers + 1, T] format\n        tokens = tokens.view(len(self.eligible_exit_layers)+1, -1)\n        shadow_confidence_scores = shadow_confidence_scores.view(len(self.eligible_exit_layers)+1, -1)\n        \n        # Update statistics (α_ℓ, τ_ℓ) and select next config\n        self.update_acceptance_and_confidence_stats(tokens, shadow_confidence_scores[:-1, :], exit_layer, context_length)\n        self.select_optimal_exit_and_gamma()\n        \n        # Clear cache for next round\n        self.clear_cached_hidden_states()\n        \n        return return_logits, return_tokens",
>   "title": "DEL.py",
>   "modified_date": "2025-07-11T23:26:20Z",
>   "url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py",
>   "display_url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py",
>   "display_title": "DEL.py",
>   "structuredContent": {
>     "title": "DEL.py",
>     "content": "import torch\nimport itertools\nfrom typing import List, Optional, Tuple\n\nfrom self_speculation.llama_model_utils import (\n    decode_next_token,\n)\n\n\nclass DEL:\n    def __init__(self, model, gamma_max, eligible_exit_layers=None, omega=1.0, gamma_min=0):\n        self.model = model\n        self.token_match_counts = torch.zeros((len(model.model.layers), 2))\n        self.confidence_sums = torch.zeros((len(model.model.layers), 2))\n        self.average_confidence = torch.zeros((len(model.model.layers), 2))\n        self.is_prefill_stage = True\n        self.omega = omega\n        if eligible_exit_layers is None:\n            self.eligible_exit_layers = list(range(0, len(self.model.model.layers)-1))\n        else:\n            self.eligible_exit_layers = eligible_exit_layers\n        self.gamma_max = gamma_max\n        self.gamma_min = gamma_min\n        self.cached_hidden_states = [[] for i in range(len(self.model.model.layers))]\n        self.current_exit_layer = self.eligible_exit_layers[0]+1\n        self.current_gamma = 1\n\n    def clear_cached_hidden_states(self):\n        \"\"\"Reset cached hidden states for all layers (used at the start of each SD round).\"\"\"\n        self.cached_hidden_states = [[] for i in range(len(self.model.model.layers))]\n    \n    def remove_old_hidden_states(self, input_length):\n        \"\"\"\n        Trim outdated cached hidden states based on consumed input length.\n        Applies a 32-token prefill adjustment on first use.\n        \"\"\"\n        if self.is_prefill_stage:\n            input_length = max(0, input_length - 32)\n            self.is_prefill_stage = False\n        if input_length == 0:\n            return 0\n        \n        for i in self.eligible_exit_layers:\n            rmln = input_length\n            if self.cached_hidden_states[i]:\n                new_hidden_states = []\n                for j in range(len(self.cached_hidden_states[i])):\n                    if rmln == 0:\n                        new_hidden_states.append(self.cached_hidden_states[i][j])\n                    elif rmln >= self.cached_hidden_states[i][j].shape[1]:\n                        rmln -= self.cached_hidden_states[i][j].shape[1]\n                    else:\n                        new_hidden_states.append(self.cached_hidden_states[i][j][:, rmln:, :])\n                        rmln = 0\n                self.cached_hidden_states[i] = new_hidden_states\n        return input_length\n\n    def update_acceptance_and_confidence_stats(self, tokens, shadow_confidence_scores, exit_layer, context_length=0):\n        \"\"\"\n        Update acceptance counts (α_ℓ) and confidence score statistics (τ_ℓ) \n        using shadow tokens from each candidate draft layer.\n        \"\"\"\n        # All but the last row are predictions from candidate draft layers.\n        layers_tokens = tokens[:-1, :]\n        \n        # Last row is the target model’s predictions (full-depth layer).\n        correct_tokens = tokens[-1, :].unsqueeze(0).expand_as(layers_tokens)\n        \n        # Boolean table indicating where layer predictions match the target model\n        token_match_table = layers_tokens == correct_tokens\n\n        # Split context tokens if present\n        if context_length > 0:\n            context_token_match_table = token_match_table[:, :context_length]\n            token_match_table = token_match_table[:, context_length:]\n            context_shadow_confidence_scores = shadow_confidence_scores[:, :context_length]\n            shadow_confidence_scores = shadow_confidence_scores[:, context_length:]\n        \n\n        # Clip tokens after the first mismatch at the current exit layer\n        if self.current_gamma > 0:\n            current_can_layer = exit_layer - 1 - self.eligible_exit_layers[0]\n            current_can_layer_false_indices = (token_match_table[current_can_layer, :] == False).nonzero(as_tuple=True)[0]\n            if len(current_can_layer_false_indices) > 0:\n                token_match_table = token_match_table[:, :current_can_layer_false_indices[0].item()+1]\n                shadow_confidence_scores = shadow_confidence_scores[:, :current_can_layer_false_indices[0].item()+1]\n        \n        # Restore context if it was separated\n        if context_length > 0:\n            token_match_table = torch.cat((context_token_match_table, token_match_table), dim=1)\n            shadow_confidence_scores = torch.cat((context_shadow_confidence_scores, shadow_confidence_scores), dim=1)\n        \n        # Sum confidence scores over matched and mismatched tokens\n        true_shadow_confidence_scores = (shadow_confidence_scores * token_match_table.float()).sum(dim=1)\n        false_shadow_confidence_scores = (shadow_confidence_scores * (~token_match_table).float()).sum(dim=1)\n        confidence_score_sums_update = torch.stack((true_shadow_confidence_scores, false_shadow_confidence_scores), dim=1)\n        \n        # Apply exponential moving average update for confidence scores\n        self.confidence_sums[self.eligible_exit_layers, :] = (\n            self.omega * self.confidence_sums[self.eligible_exit_layers, :] + confidence_score_sums_update.to(self.confidence_sums)\n        )\n        \n        # Count number of matches/mismatches per layer\n        true_counts = token_match_table.sum(dim=1)\n        false_counts = token_match_table.shape[1]-true_counts\n        match_counts_update = torch.stack((true_counts, false_counts), dim=1)\n        \n        # Update moving average for token matches\n        self.token_match_counts[self.eligible_exit_layers, :] = (\n            self.omega * self.token_match_counts[self.eligible_exit_layers, :] + match_counts_update.to(self.token_match_counts)\n        )\n        \n        # Derive average confidence per layer over matched/mismatched tokens\n        self.average_confidence = self.confidence_sums / self.token_match_counts\n    \n    def select_optimal_exit_and_gamma(self):\n        \"\"\"\n        Compute TPL(ℓ, γ) for all candidate exit layers and speculation lengths,\n        and select the configuration that maximizes it.\n        \"\"\"\n        \n        # Estimate α_ℓ = accepted / total tokens for each candidate layer\n        probs = self.token_match_counts[self.eligible_exit_layers, 0] / torch.sum(self.token_match_counts[self.eligible_exit_layers, :], dim=1)\n        probs = torch.where(torch.isnan(probs), torch.tensor(float('-inf')), probs)\n        \n        # Compute expected number of accepted tokens per speculation length using α_ℓ\n        gamma_probs = torch.zeros((len(self.eligible_exit_layers), self.gamma_max))\n        for i in range(self.gamma_max):\n            if i == 0:\n                gamma_probs[:, i] = 1\n            else:\n                gamma_probs[:, i] = 1 + probs * gamma_probs[:, i-1]\n        \n        # Cost model: each round costs γ * draft_cost + verification_cost\n        draft_cost = torch.tensor(self.eligible_exit_layers) + 1\n        verify_cost = len(self.model.model.layers) - draft_cost\n        layers_costs = torch.zeros((len(self.eligible_exit_layers), self.gamma_max))\n        for i in range(self.gamma_max):\n            layers_costs[:, i] = verify_cost + (i+1) * draft_cost\n        \n        # Compute benefit = expected tokens / total cost = TPL\n        tpl = gamma_probs / layers_costs\n        \n        # Enforce minimum speculation length\n        if self.gamma_min > 0:\n            tpl[:, :self.gamma_min] = float('-inf')\n            \n        # Select (layer, γ) pair that maximizes TPL\n        del_layer, del_gamma = divmod(torch.argmax(tpl).item(), tpl.shape[1])\n        \n        # Convert back to global layer index\n        self.current_exit_layer = del_layer + self.eligible_exit_layers[0] + 1\n        self.current_gamma = del_gamma\n\n    def run(self, exit_layer, input_length, sample):\n        \"\"\"\n        Finalize verification of the current speculative decoding round:\n        - Compute shadow logits\n        - Derive token match and confidence stats\n        - Update optimal exit layer and speculation length (E, γ)\n        \"\"\"\n        \n        # Compute number of new tokens generated\n        output_length = self.cached_hidden_states[-1][0].shape[1] - input_length\n        \n        # Remove past tokens (prefill or already processed)\n        trim_length = self.remove_old_hidden_states(input_length)\n        context_length = input_length - trim_length\n\n        # Flatten cached hidden states from all layers\n        flattened_hiddens = list(itertools.chain.from_iterable(self.cached_hidden_states))\n        if not flattened_hiddens:\n            return None, None\n        \n        # Send all to LM-head (on correct device)\n        lm_head_device = self.model.lm_head.weight.device\n        flattened_hiddens = [h.to(lm_head_device) for h in flattened_hiddens]\n        hiddens = torch.cat(flattened_hiddens, dim=1)\n        hiddens = self.model.model.norm(hiddens)\n        \n        # Compute logits and extract segment of interest\n        logits = self.model.lm_head(hiddens)\n        return_logits = logits[:, -(input_length+output_length):, :]\n        logits = torch.cat(\n            (logits[:, 0:-(input_length + output_length), :], logits[:, -(context_length + output_length):, :]),\n            dim=1\n        )\n        \n        # Greedy decode shadow tokens and extract their confidence scores\n        tokens, shadow_confidence_scores = decode_next_token(logits=logits, sample=False)\n        shadow_confidence_scores = shadow_confidence_scores[torch.arange(shadow_confidence_scores.size(0)), tokens.squeeze(0)]\n        \n        # Only return generated tokens (exclude context) if not sampling\n        return_tokens = tokens[:, -output_length:] if not sample else None\n        \n        # Reshape to [#layers + 1, T] format\n        tokens = tokens.view(len(self.eligible_exit_layers)+1, -1)\n        shadow_confidence_scores = shadow_confidence_scores.view(len(self.eligible_exit_layers)+1, -1)\n        \n        # Update statistics (α_ℓ, τ_ℓ) and select next config\n        self.update_acceptance_and_confidence_stats(tokens, shadow_confidence_scores[:-1, :], exit_layer, context_length)\n        self.select_optimal_exit_and_gamma()\n        \n        # Clear cache for next round\n        self.clear_cached_hidden_states()\n        \n        return return_logits, return_tokens",
>     "display_url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL.py"
>   }
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch
>
> ```json
> {
>   "url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n# All rights reserved.\n#\n# This source code is licensed under the license found in the\n# LICENSE file in the root directory of this source tree.\n#\n\nfrom typing import List, Optional, Tuple\n\nimport colorama\nimport torch\nimport itertools\nimport matplotlib.pyplot as plt\nimport os\n\nimport transformers\nfrom self_speculation.generator_base import (\n    GenerationConfig,\n    GenerationStrategy,\n    GenerationStrategyResult,\n)\nfrom self_speculation.speculative_streamer import SpeculativeTextStreamer\nfrom self_speculation.llama_model_utils import (\n    crop_past_key_values,\n    decode_next_token,\n    forward_early_DEL,\n    forward_remainder_DEL,\n)\nfrom self_speculation.DEL import DEL\n\n\n\ndef max_fn(x, eps=1e-6):\n    x_max = torch.where(x > 0, x, 0)\n    return x_max / (torch.sum(x_max) + eps)\n\n\nclass DELSpeculativeGenerationStrategy(GenerationStrategy):\n    def __init__(self):\n        super().__init__()\n        self.DEL = None\n    \n    def generate_token_ids(\n        self,\n        model: transformers.LlamaForCausalLM,\n        input_ids: List[int],\n        eos_token_ids: List[int],\n        generation_config: GenerationConfig,\n        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,\n        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,\n        streamer: Optional[transformers.TextStreamer] = None,\n    ) -> GenerationStrategyResult:\n        past_key_values = None\n        input_ids_list = input_ids\n        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)\n        output_ids: List[int] = []\n\n        # DEL Module Definition\n        self.DEL = DEL(model, gamma_max=18, omega=0.95) \n       \n        # DEL with fixed exit_layer\n        # self.DEL = DEL(model, eligible_exit_layers=list(range(generation_config.exit_layer-1, generation_config.exit_layer)), gamma_max=18, omega=0.95) \n        \n        # DEL with limited exit_layer range\n        # self.DEL = DEL(model, eligible_exit_layers=list(range(3, len(model.model.layers)-1)), gamma_max=18, omega = 0.95) \n        \n\n        calls: int = 0\n        total_draft_matches = 0\n        total_generations = 0\n        total_layers = 0\n        total_tokens = 0\n        while len(output_ids) < generation_config.max_steps:\n            current_exit_layer = self.DEL.current_exit_layer\n            (\n                input_ids,\n                output_ids,\n                past_key_values,\n                number_of_matches,\n                num_speculations,\n            ) = self.single_step_speculation(\n                model=model,\n                input_ids_list=input_ids_list,\n                input_ids=input_ids,\n                output_ids=output_ids,\n                num_speculations=min(\n                    self.DEL.current_gamma,\n                    generation_config.max_steps - len(output_ids) - 1,\n                ),\n                past_key_values=past_key_values,\n                exit_layer=self.DEL.current_exit_layer,\n                eos_token_ids=eos_token_ids,\n                calls=calls,\n                sample=generation_config.sample,\n                temperature=generation_config.temperature,\n                top_k=generation_config.top_k,\n                top_p=generation_config.top_p,\n                logits_processors=logits_processors,\n                stopping_criteria=stopping_criteria,\n                streamer=streamer,\n            )\n            calls += 1\n            total_draft_matches += number_of_matches\n            total_generations += num_speculations\n            total_tokens += (number_of_matches+1)\n            total_layers += current_exit_layer * num_speculations + len(model.model.layers)\n            \n            eos_found = False\n            for eos_token_id in eos_token_ids:\n                if eos_token_id in output_ids:\n                    # break out of loop when we get an EOS token\n                    # remove the EOS token id\n                    output_ids = output_ids[: output_ids.index(eos_token_id)]\n                    eos_found = True\n                    break\n            if eos_found:\n                break\n            if stopping_criteria:\n                # TODO: when implementing batch size > 1, stop each sample separately?\n                if torch.all(stopping_criteria(input_ids, scores=None)):\n                    break\n        \n        return GenerationStrategyResult(\n            predicted_tokens=output_ids,\n            acceptance_rate=total_draft_matches / total_generations if total_generations > 0 else 0,\n            tokens_per_layer=total_tokens / total_layers if total_layers > 0 else 0,\n        )\n\n    # TODO: remove calls, input_ids_list, rely on generation config\n    def single_step_speculation(\n        self,\n        model: transformers.LlamaForCausalLM,\n        input_ids: torch.Tensor,\n        input_ids_list: List[int],\n        output_ids: List[int],\n        num_speculations: int,\n        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],\n        eos_token_ids: List[int],\n        calls: int,\n        exit_layer: int,\n        sample: Optional[bool] = False,\n        temperature: Optional[float] = 0.7,\n        top_k: Optional[int] = 50,\n        top_p: Optional[float] = 0.95,\n        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,\n        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,\n        streamer: Optional[transformers.TextStreamer] = None\n    ):\n        prompt_length: int = input_ids.size(1)\n        draft_input_ids = input_ids.clone()\n        draft_output_ids: List[int] = []\n        if sample:\n            draft_probabilities: List[torch.Tensor] = []\n        exit_query_cache = None\n\n        d_max = 1 if self.DEL.is_prefill_stage else 18\n        for di in range(d_max):\n            draft_result = forward_early_DEL(\n                model,\n                draft_input_ids,\n                past_key_values,\n                exit_layer,\n                exit_query_cache,\n                self.DEL,\n            )\n            past_key_values = draft_result.past_key_values\n            exit_query_cache = draft_result.exit_query_cache\n            draft_logits = draft_result.logits\n            if logits_processors:\n                draft_logits = logits_processors(draft_input_ids, draft_logits)\n\n            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)\n            draft_next_token = draft_next_token.item()\n            draft_output_ids.append(draft_next_token)\n            if sample:\n                draft_probabilities.append(draft_next_prob)\n            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)\n            if draft_next_token in eos_token_ids:\n                # break out of loop when we get an EOS token\n                break\n\n            # Get DEL threshold τₗ for this exit layer\n            draft_token_confidence_score = draft_next_prob[0, draft_next_token].item()\n            avg_confidence_stats = self.DEL.average_confidence[exit_layer - 1]  # [τ_match, τ_mismatch]\n            tau_threshold_for_exit_layer = ((avg_confidence_stats[0] + avg_confidence_stats[1]) / 2).item()\n\n            if (not self.DEL.is_prefill_stage) and draft_token_confidence_score < tau_threshold_for_exit_layer:\n                break\n                \n        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make\n        # 1 x (T_d  + T_p)\n        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)\n        prefill_token_ids = torch.cat(\n            [input_ids, draft_output_ids],\n            dim=-1,\n        )\n\n        if streamer:\n            if isinstance(streamer, SpeculativeTextStreamer):\n                print(colorama.Fore.LIGHTMAGENTA_EX, end=\"\")\n                streamer.put(draft_output_ids, is_draft=True)\n\n        # logits: 1 x (T_d  + T_p) x V\n        verify_results = forward_remainder_DEL(\n            model,\n            prefill_token_ids.int(),\n            past_key_values,\n            exit_layer,\n            exit_query_cache,\n            self.DEL,\n        )\n        logits, del_tokens = self.DEL.run(exit_layer, prompt_length-1, sample)\n\n        if logits_processors:\n            logits = logits_processors(prefill_token_ids, logits)\n        past_key_values = verify_results.past_key_values\n        # only select the logits relevant to what the draft has outputted.\n        # verification_logits: 1 x T_d x V\n        verification_logits = logits[:, prompt_length - 1 :, :]\n\n        # verified_tokens: 1 x (T_d)\n        # There is a predicted token for every token in the draft output ids list, however note that the\n        # first tokens (or first N tokens) are coming from the prompt\n        if sample:\n            verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)\n        else:\n            verified_tokens = del_tokens\n            verified_probabilities = None\n\n        # skip verification of the last token as it is a new token predicted from the main model\n        verified_tokens = verified_tokens.to(prefill_token_ids)\n        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]\n\n        # number of matches is the index of the number of tokens we are accepting from the draft\n        if not sample:\n            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()\n        else:\n            number_of_matches = 0\n            rand = torch.rand_like(draft_output_ids, dtype=torch.float)\n            for i in range(draft_output_ids.numel()):\n                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):\n                    number_of_matches += 1\n                else:\n                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()\n                    break\n\n        # accept the `number_of_matches` tokens from the draft with one more from the main model\n        # since we re-use the same cachem the input id should only be the last accepted token TODO check this\n        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]\n        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())\n        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())\n\n        if streamer:\n            if isinstance(streamer, SpeculativeTextStreamer):\n                streamer.delete(len(draft_output_ids[0, :]))\n                print(colorama.Fore.GREEN, end=\"\")\n                streamer.put(draft_output_ids[0, : number_of_matches])\n                print(colorama.Style.RESET_ALL, end=\"\")\n                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])\n            else:\n                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))\n\n        # we want the entire output sequence + input sequence\n        past_key_values = crop_past_key_values(\n            past_key_values, len(input_ids_list) + len(output_ids) - 1\n        )\n        \n        return (\n            input_ids,\n            output_ids,\n            past_key_values,\n            number_of_matches,\n            draft_output_ids.numel(),\n        )\n",
>   "title": "DEL_speculation_generator.py",
>   "modified_date": "2025-07-11T23:26:20Z",
>   "url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py",
>   "display_url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py",
>   "display_title": "DEL_speculation_generator.py",
>   "structuredContent": {
>     "title": "DEL_speculation_generator.py",
>     "content": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n# All rights reserved.\n#\n# This source code is licensed under the license found in the\n# LICENSE file in the root directory of this source tree.\n#\n\nfrom typing import List, Optional, Tuple\n\nimport colorama\nimport torch\nimport itertools\nimport matplotlib.pyplot as plt\nimport os\n\nimport transformers\nfrom self_speculation.generator_base import (\n    GenerationConfig,\n    GenerationStrategy,\n    GenerationStrategyResult,\n)\nfrom self_speculation.speculative_streamer import SpeculativeTextStreamer\nfrom self_speculation.llama_model_utils import (\n    crop_past_key_values,\n    decode_next_token,\n    forward_early_DEL,\n    forward_remainder_DEL,\n)\nfrom self_speculation.DEL import DEL\n\n\n\ndef max_fn(x, eps=1e-6):\n    x_max = torch.where(x > 0, x, 0)\n    return x_max / (torch.sum(x_max) + eps)\n\n\nclass DELSpeculativeGenerationStrategy(GenerationStrategy):\n    def __init__(self):\n        super().__init__()\n        self.DEL = None\n    \n    def generate_token_ids(\n        self,\n        model: transformers.LlamaForCausalLM,\n        input_ids: List[int],\n        eos_token_ids: List[int],\n        generation_config: GenerationConfig,\n        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,\n        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,\n        streamer: Optional[transformers.TextStreamer] = None,\n    ) -> GenerationStrategyResult:\n        past_key_values = None\n        input_ids_list = input_ids\n        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)\n        output_ids: List[int] = []\n\n        # DEL Module Definition\n        self.DEL = DEL(model, gamma_max=18, omega=0.95) \n       \n        # DEL with fixed exit_layer\n        # self.DEL = DEL(model, eligible_exit_layers=list(range(generation_config.exit_layer-1, generation_config.exit_layer)), gamma_max=18, omega=0.95) \n        \n        # DEL with limited exit_layer range\n        # self.DEL = DEL(model, eligible_exit_layers=list(range(3, len(model.model.layers)-1)), gamma_max=18, omega = 0.95) \n        \n\n        calls: int = 0\n        total_draft_matches = 0\n        total_generations = 0\n        total_layers = 0\n        total_tokens = 0\n        while len(output_ids) < generation_config.max_steps:\n            current_exit_layer = self.DEL.current_exit_layer\n            (\n                input_ids,\n                output_ids,\n                past_key_values,\n                number_of_matches,\n                num_speculations,\n            ) = self.single_step_speculation(\n                model=model,\n                input_ids_list=input_ids_list,\n                input_ids=input_ids,\n                output_ids=output_ids,\n                num_speculations=min(\n                    self.DEL.current_gamma,\n                    generation_config.max_steps - len(output_ids) - 1,\n                ),\n                past_key_values=past_key_values,\n                exit_layer=self.DEL.current_exit_layer,\n                eos_token_ids=eos_token_ids,\n                calls=calls,\n                sample=generation_config.sample,\n                temperature=generation_config.temperature,\n                top_k=generation_config.top_k,\n                top_p=generation_config.top_p,\n                logits_processors=logits_processors,\n                stopping_criteria=stopping_criteria,\n                streamer=streamer,\n            )\n            calls += 1\n            total_draft_matches += number_of_matches\n            total_generations += num_speculations\n            total_tokens += (number_of_matches+1)\n            total_layers += current_exit_layer * num_speculations + len(model.model.layers)\n            \n            eos_found = False\n            for eos_token_id in eos_token_ids:\n                if eos_token_id in output_ids:\n                    # break out of loop when we get an EOS token\n                    # remove the EOS token id\n                    output_ids = output_ids[: output_ids.index(eos_token_id)]\n                    eos_found = True\n                    break\n            if eos_found:\n                break\n            if stopping_criteria:\n                # TODO: when implementing batch size > 1, stop each sample separately?\n                if torch.all(stopping_criteria(input_ids, scores=None)):\n                    break\n        \n        return GenerationStrategyResult(\n            predicted_tokens=output_ids,\n            acceptance_rate=total_draft_matches / total_generations if total_generations > 0 else 0,\n            tokens_per_layer=total_tokens / total_layers if total_layers > 0 else 0,\n        )\n\n    # TODO: remove calls, input_ids_list, rely on generation config\n    def single_step_speculation(\n        self,\n        model: transformers.LlamaForCausalLM,\n        input_ids: torch.Tensor,\n        input_ids_list: List[int],\n        output_ids: List[int],\n        num_speculations: int,\n        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],\n        eos_token_ids: List[int],\n        calls: int,\n        exit_layer: int,\n        sample: Optional[bool] = False,\n        temperature: Optional[float] = 0.7,\n        top_k: Optional[int] = 50,\n        top_p: Optional[float] = 0.95,\n        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,\n        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,\n        streamer: Optional[transformers.TextStreamer] = None\n    ):\n        prompt_length: int = input_ids.size(1)\n        draft_input_ids = input_ids.clone()\n        draft_output_ids: List[int] = []\n        if sample:\n            draft_probabilities: List[torch.Tensor] = []\n        exit_query_cache = None\n\n        d_max = 1 if self.DEL.is_prefill_stage else 18\n        for di in range(d_max):\n            draft_result = forward_early_DEL(\n                model,\n                draft_input_ids,\n                past_key_values,\n                exit_layer,\n                exit_query_cache,\n                self.DEL,\n            )\n            past_key_values = draft_result.past_key_values\n            exit_query_cache = draft_result.exit_query_cache\n            draft_logits = draft_result.logits\n            if logits_processors:\n                draft_logits = logits_processors(draft_input_ids, draft_logits)\n\n            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)\n            draft_next_token = draft_next_token.item()\n            draft_output_ids.append(draft_next_token)\n            if sample:\n                draft_probabilities.append(draft_next_prob)\n            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)\n            if draft_next_token in eos_token_ids:\n                # break out of loop when we get an EOS token\n                break\n\n            # Get DEL threshold τₗ for this exit layer\n            draft_token_confidence_score = draft_next_prob[0, draft_next_token].item()\n            avg_confidence_stats = self.DEL.average_confidence[exit_layer - 1]  # [τ_match, τ_mismatch]\n            tau_threshold_for_exit_layer = ((avg_confidence_stats[0] + avg_confidence_stats[1]) / 2).item()\n\n            if (not self.DEL.is_prefill_stage) and draft_token_confidence_score < tau_threshold_for_exit_layer:\n                break\n                \n        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make\n        # 1 x (T_d  + T_p)\n        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)\n        prefill_token_ids = torch.cat(\n            [input_ids, draft_output_ids],\n            dim=-1,\n        )\n\n        if streamer:\n            if isinstance(streamer, SpeculativeTextStreamer):\n                print(colorama.Fore.LIGHTMAGENTA_EX, end=\"\")\n                streamer.put(draft_output_ids, is_draft=True)\n\n        # logits: 1 x (T_d  + T_p) x V\n        verify_results = forward_remainder_DEL(\n            model,\n            prefill_token_ids.int(),\n            past_key_values,\n            exit_layer,\n            exit_query_cache,\n            self.DEL,\n        )\n        logits, del_tokens = self.DEL.run(exit_layer, prompt_length-1, sample)\n\n        if logits_processors:\n            logits = logits_processors(prefill_token_ids, logits)\n        past_key_values = verify_results.past_key_values\n        # only select the logits relevant to what the draft has outputted.\n        # verification_logits: 1 x T_d x V\n        verification_logits = logits[:, prompt_length - 1 :, :]\n\n        # verified_tokens: 1 x (T_d)\n        # There is a predicted token for every token in the draft output ids list, however note that the\n        # first tokens (or first N tokens) are coming from the prompt\n        if sample:\n            verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)\n        else:\n            verified_tokens = del_tokens\n            verified_probabilities = None\n\n        # skip verification of the last token as it is a new token predicted from the main model\n        verified_tokens = verified_tokens.to(prefill_token_ids)\n        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]\n\n        # number of matches is the index of the number of tokens we are accepting from the draft\n        if not sample:\n            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()\n        else:\n            number_of_matches = 0\n            rand = torch.rand_like(draft_output_ids, dtype=torch.float)\n            for i in range(draft_output_ids.numel()):\n                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):\n                    number_of_matches += 1\n                else:\n                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()\n                    break\n\n        # accept the `number_of_matches` tokens from the draft with one more from the main model\n        # since we re-use the same cachem the input id should only be the last accepted token TODO check this\n        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]\n        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())\n        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())\n\n        if streamer:\n            if isinstance(streamer, SpeculativeTextStreamer):\n                streamer.delete(len(draft_output_ids[0, :]))\n                print(colorama.Fore.GREEN, end=\"\")\n                streamer.put(draft_output_ids[0, : number_of_matches])\n                print(colorama.Style.RESET_ALL, end=\"\")\n                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])\n            else:\n                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))\n\n        # we want the entire output sequence + input sequence\n        past_key_values = crop_past_key_values(\n            past_key_values, len(input_ids_list) + len(output_ids) - 1\n        )\n        \n        return (\n            input_ids,\n            output_ids,\n            past_key_values,\n            number_of_matches,\n            draft_output_ids.numel(),\n        )\n",
>     "display_url": "https://github.com/hoenza/DEL/blob/main/self_speculation/DEL_speculation_generator.py"
>   }
> }
> ```
>
> MCP tool call
>
> codex_apps.github.fetch
>
> ```json
> {
>   "url": "https://github.com/hoenza/DEL"
> }
> ```
>
> ```text
> Action completed.
> ```
>
> ```json
> {
>   "content": "{\"id\":956280990,\"node_id\":\"R_kgDOOP-wng\",\"name\":\"DEL\",\"full_name\":\"hoenza/DEL\",\"private\":false,\"owner\":{\"login\":\"hoenza\",\"id\":37626770,\"node_id\":\"MDQ6VXNlcjM3NjI2Nzcw\",\"avatar_url\":\"https://avatars.githubusercontent.com/u/37626770?v=4\",\"gravatar_id\":\"\",\"url\":\"https://api.github.com/users/hoenza\",\"html_url\":\"https://github.com/hoenza\",\"followers_url\":\"https://api.github.com/users/hoenza/followers\",\"following_url\":\"https://api.github.com/users/hoenza/following{/other_user}\",\"gists_url\":\"https://api.github.com/users/hoenza/gists{/gist_id}\",\"starred_url\":\"https://api.github.com/users/hoenza/starred{/owner}{/repo}\",\"subscriptions_url\":\"https://api.github.com/users/hoenza/subscriptions\",\"organizations_url\":\"https://api.github.com/users/hoenza/orgs\",\"repos_url\":\"https://api.github.com/users/hoenza/repos\",\"events_url\":\"https://api.github.com/users/hoenza/events{/privacy}\",\"received_events_url\":\"https://api.github.com/users/hoenza/received_events\",\"type\":\"User\",\"user_view_type\":\"public\",\"site_admin\":false},\"html_url\":\"https://github.com/hoenza/DEL\",\"description\":\"[COLM 2025] DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding\",\"fork\":false,\"url\":\"https://api.github.com/repos/hoenza/DEL\",\"forks_url\":\"https://api.github.com/repos/hoenza/DEL/forks\",\"keys_url\":\"https://api.github.com/repos/hoenza/DEL/keys{/key_id}\",\"collaborators_url\":\"https://api.github.com/repos/hoenza/DEL/collaborators{/collaborator}\",\"teams_url\":\"https://api.github.com/repos/hoenza/DEL/teams\",\"hooks_url\":\"https://api.github.com/repos/hoenza/DEL/hooks\",\"issue_events_url\":\"https://api.github.com/repos/hoenza/DEL/issues/events{/number}\",\"events_url\":\"https://api.github.com/repos/hoenza/DEL/events\",\"assignees_url\":\"https://api.github.com/repos/hoenza/DEL/assignees{/user}\",\"branches_url\":\"https://api.github.com/repos/hoenza/DEL/branches{/branch}\",\"tags_url\":\"https://api.github.com/repos/hoenza/DEL/tags\",\"blobs_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs{/sha}\",\"git_tags_url\":\"https://api.github.com/repos/hoenza/DEL/git/tags{/sha}\",\"git_refs_url\":\"https://api.github.com/repos/hoenza/DEL/git/refs{/sha}\",\"trees_url\":\"https://api.github.com/repos/hoenza/DEL/git/trees{/sha}\",\"statuses_url\":\"https://api.github.com/repos/hoenza/DEL/statuses/{sha}\",\"languages_url\":\"https://api.github.com/repos/hoenza/DEL/languages\",\"stargazers_url\":\"https://api.github.com/repos/hoenza/DEL/stargazers\",\"contributors_url\":\"https://api.github.com/repos/hoenza/DEL/contributors\",\"subscribers_url\":\"https://api.github.com/repos/hoenza/DEL/subscribers\",\"subscription_url\":\"https://api.github.com/repos/hoenza/DEL/subscription\",\"commits_url\":\"https://api.github.com/repos/hoenza/DEL/commits{/sha}\",\"git_commits_url\":\"https://api.github.com/repos/hoenza/DEL/git/commits{/sha}\",\"comments_url\":\"https://api.github.com/repos/hoenza/DEL/comments{/number}\",\"issue_comment_url\":\"https://api.github.com/repos/hoenza/DEL/issues/comments{/number}\",\"contents_url\":\"https://api.github.com/repos/hoenza/DEL/contents/{+path}\",\"compare_url\":\"https://api.github.com/repos/hoenza/DEL/compare/{base}...{head}\",\"merges_url\":\"https://api.github.com/repos/hoenza/DEL/merges\",\"archive_url\":\"https://api.github.com/repos/hoenza/DEL/{archive_format}{/ref}\",\"downloads_url\":\"https://api.github.com/repos/hoenza/DEL/downloads\",\"issues_url\":\"https://api.github.com/repos/hoenza/DEL/issues{/number}\",\"pulls_url\":\"https://api.github.com/repos/hoenza/DEL/pulls{/number}\",\"milestones_url\":\"https://api.github.com/repos/hoenza/DEL/milestones{/number}\",\"notifications_url\":\"https://api.github.com/repos/hoenza/DEL/notifications{?since,all,participating}\",\"labels_url\":\"https://api.github.com/repos/hoenza/DEL/labels{/name}\",\"releases_url\":\"https://api.github.com/repos/hoenza/DEL/releases{/id}\",\"deployments_url\":\"https://api.github.com/repos/hoenza/DEL/deployments\",\"created_at\":\"2025-03-28T01:53:19Z\",\"updated_at\":\"2026-07-30T02:12:39Z\",\"pushed_at\":\"2026-05-11T21:05:38Z\",\"git_url\":\"git://github.com/hoenza/DEL.git\",\"ssh_url\":\"git@github.com:hoenza/DEL.git\",\"clone_url\":\"https://github.com/hoenza/DEL.git\",\"svn_url\":\"https://github.com/hoenza/DEL\",\"homepage\":\"\",\"size\":3657,\"stargazers_count\":11,\"watchers_count\":11,\"language\":\"Python\",\"has_issues\":true,\"has_projects\":true,\"has_downloads\":false,\"has_wiki\":true,\"has_pages\":false,\"has_discussions\":false,\"forks_count\":2,\"mirror_url\":null,\"archived\":false,\"disabled\":false,\"open_issues_count\":0,\"license\":null,\"allow_forking\":true,\"is_template\":false,\"web_commit_signoff_required\":false,\"has_pull_requests\":true,\"pull_request_creation_policy\":\"all\",\"topics\":[\"speculative-decoding\"],\"visibility\":\"public\",\"forks\":2,\"open_issues\":0,\"watchers\":11,\"default_branch\":\"main\",\"permissions\":{\"admin\":false,\"maintain\":false,\"push\":false,\"triage\":false,\"pull\":true},\"network_count\":2,\"subscribers_count\":0}",
>   "title": "DEL",
>   "modified_date": null,
>   "url": "https://github.com/hoenza/DEL",
>   "display_url": "https://github.com/hoenza/DEL",
>   "display_title": "DEL",
>   "structuredContent": {
>     "title": "DEL",
>     "content": "{\"id\":956280990,\"node_id\":\"R_kgDOOP-wng\",\"name\":\"DEL\",\"full_name\":\"hoenza/DEL\",\"private\":false,\"owner\":{\"login\":\"hoenza\",\"id\":37626770,\"node_id\":\"MDQ6VXNlcjM3NjI2Nzcw\",\"avatar_url\":\"https://avatars.githubusercontent.com/u/37626770?v=4\",\"gravatar_id\":\"\",\"url\":\"https://api.github.com/users/hoenza\",\"html_url\":\"https://github.com/hoenza\",\"followers_url\":\"https://api.github.com/users/hoenza/followers\",\"following_url\":\"https://api.github.com/users/hoenza/following{/other_user}\",\"gists_url\":\"https://api.github.com/users/hoenza/gists{/gist_id}\",\"starred_url\":\"https://api.github.com/users/hoenza/starred{/owner}{/repo}\",\"subscriptions_url\":\"https://api.github.com/users/hoenza/subscriptions\",\"organizations_url\":\"https://api.github.com/users/hoenza/orgs\",\"repos_url\":\"https://api.github.com/users/hoenza/repos\",\"events_url\":\"https://api.github.com/users/hoenza/events{/privacy}\",\"received_events_url\":\"https://api.github.com/users/hoenza/received_events\",\"type\":\"User\",\"user_view_type\":\"public\",\"site_admin\":false},\"html_url\":\"https://github.com/hoenza/DEL\",\"description\":\"[COLM 2025] DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding\",\"fork\":false,\"url\":\"https://api.github.com/repos/hoenza/DEL\",\"forks_url\":\"https://api.github.com/repos/hoenza/DEL/forks\",\"keys_url\":\"https://api.github.com/repos/hoenza/DEL/keys{/key_id}\",\"collaborators_url\":\"https://api.github.com/repos/hoenza/DEL/collaborators{/collaborator}\",\"teams_url\":\"https://api.github.com/repos/hoenza/DEL/teams\",\"hooks_url\":\"https://api.github.com/repos/hoenza/DEL/hooks\",\"issue_events_url\":\"https://api.github.com/repos/hoenza/DEL/issues/events{/number}\",\"events_url\":\"https://api.github.com/repos/hoenza/DEL/events\",\"assignees_url\":\"https://api.github.com/repos/hoenza/DEL/assignees{/user}\",\"branches_url\":\"https://api.github.com/repos/hoenza/DEL/branches{/branch}\",\"tags_url\":\"https://api.github.com/repos/hoenza/DEL/tags\",\"blobs_url\":\"https://api.github.com/repos/hoenza/DEL/git/blobs{/sha}\",\"git_tags_url\":\"https://api.github.com/repos/hoenza/DEL/git/tags{/sha}\",\"git_refs_url\":\"https://api.github.com/repos/hoenza/DEL/git/refs{/sha}\",\"trees_url\":\"https://api.github.com/repos/hoenza/DEL/git/trees{/sha}\",\"statuses_url\":\"https://api.github.com/repos/hoenza/DEL/statuses/{sha}\",\"languages_url\":\"https://api.github.com/repos/hoenza/DEL/languages\",\"stargazers_url\":\"https://api.github.com/repos/hoenza/DEL/stargazers\",\"contributors_url\":\"https://api.github.com/repos/hoenza/DEL/contributors\",\"subscribers_url\":\"https://api.github.com/repos/hoenza/DEL/subscribers\",\"subscription_url\":\"https://api.github.com/repos/hoenza/DEL/subscription\",\"commits_url\":\"https://api.github.com/repos/hoenza/DEL/commits{/sha}\",\"git_commits_url\":\"https://api.github.com/repos/hoenza/DEL/git/commits{/sha}\",\"comments_url\":\"https://api.github.com/repos/hoenza/DEL/comments{/number}\",\"issue_comment_url\":\"https://api.github.com/repos/hoenza/DEL/issues/comments{/number}\",\"contents_url\":\"https://api.github.com/repos/hoenza/DEL/contents/{+path}\",\"compare_url\":\"https://api.github.com/repos/hoenza/DEL/compare/{base}...{head}\",\"merges_url\":\"https://api.github.com/repos/hoenza/DEL/merges\",\"archive_url\":\"https://api.github.com/repos/hoenza/DEL/{archive_format}{/ref}\",\"downloads_url\":\"https://api.github.com/repos/hoenza/DEL/downloads\",\"issues_url\":\"https://api.github.com/repos/hoenza/DEL/issues{/number}\",\"pulls_url\":\"https://api.github.com/repos/hoenza/DEL/pulls{/number}\",\"milestones_url\":\"https://api.github.com/repos/hoenza/DEL/milestones{/number}\",\"notifications_url\":\"https://api.github.com/repos/hoenza/DEL/notifications{?since,all,participating}\",\"labels_url\":\"https://api.github.com/repos/hoenza/DEL/labels{/name}\",\"releases_url\":\"https://api.github.com/repos/hoenza/DEL/releases{/id}\",\"deployments_url\":\"https://api.github.com/repos/hoenza/DEL/deployments\",\"created_at\":\"2025-03-28T01:53:19Z\",\"updated_at\":\"2026-07-30T02:12:39Z\",\"pushed_at\":\"2026-05-11T21:05:38Z\",\"git_url\":\"git://github.com/hoenza/DEL.git\",\"ssh_url\":\"git@github.com:hoenza/DEL.git\",\"clone_url\":\"https://github.com/hoenza/DEL.git\",\"svn_url\":\"https://github.com/hoenza/DEL\",\"homepage\":\"\",\"size\":3657,\"stargazers_count\":11,\"watchers_count\":11,\"language\":\"Python\",\"has_issues\":true,\"has_projects\":true,\"has_downloads\":false,\"has_wiki\":true,\"has_pages\":false,\"has_discussions\":false,\"forks_count\":2,\"mirror_url\":null,\"archived\":false,\"disabled\":false,\"open_issues_count\":0,\"license\":null,\"allow_forking\":true,\"is_template\":false,\"web_commit_signoff_required\":false,\"has_pull_requests\":true,\"pull_request_creation_policy\":\"all\",\"topics\":[\"speculative-decoding\"],\"visibility\":\"public\",\"forks\":2,\"open_issues\":0,\"watchers\":11,\"default_branch\":\"main\",\"permissions\":{\"admin\":false,\"maintain\":false,\"push\":false,\"triage\":false,\"pull\":true},\"network_count\":2,\"subscribers_count\":0}",
>     "display_url": "https://github.com/hoenza/DEL"
>   }
> }
> ```
>
> </details>
>
> DEL’s public code and paper differ in a few details, including draft stopping and length bounds. The repository also has no declared top-level license. I’ll implement the published method independently, cite it, and document the choices rather than copy the upstream code.
>
> I’ll also add fixed depth 12 / length 3 to the comparison. That will let us distinguish the value of adaptation from the value of choosing a good fixed setting.
>
> <details><summary>Created 3 files, edited a file, ran 4 commands</summary>
>
> - Ran `Get-Content hardware_aware_selective_feedback/run.py | Select-Object -Skip 338 -First 270; Get-Content hardware_aware_selective_feedback/engine.py | Select-Object -Skip 385 -First 70; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader; python -c "import torch, transformers; print(torch.__version__,transformers.__version__)"`
> - Ran `Get-Content hardware_aware_selective_feedback/measurement.py -TotalCount 100; Get-Content hardware_aware_selective_feedback/tests/test_engine.py | Select-Object -Skip 110 -First 90`
> - Wrote `./hardware_aware_selective_feedback/del_policy.py` (+216 -0)
> - Wrote `./hardware_aware_selective_feedback/del_comparison.py` (+69 -0)
> - Wrote `./hardware_aware_selective_feedback/run.py` (+66 -9)
> - Wrote `./hardware_aware_selective_feedback/tests/test_del.py` (+162 -0)
> - Ran `python -m unittest hardware_aware_selective_feedback.tests.test_del -v`
> - Ran `python -m hardware_aware_selective_feedback.run --probe --device cuda; git diff --check`
>
> </details>
>
> The DEL baseline is implemented: it initializes from prompt feedback, chooses an exit each round, stops drafts using confidence, and updates statistics only on valid prefixes. The comparison adds two policies—DEL and fixed depth 12—while leaving the existing seven-policy run unchanged by default.
>
> I’m testing changing-depth cache behavior, rejected drafts, EOS handling, and short output limits before running it on the GPU.
>
> <details><summary>Created a file, edited a file, ran 5 commands</summary>
>
> - Ran `python -m unittest discover -s hardware_aware_selective_feedback/tests -v`
> - Ran `.\.venv\Scripts\python.exe -m unittest hardware_aware_selective_feedback.tests.test_del -v`
> - Wrote `./README.md` (+6 -0)
> - Wrote `./hardware_aware_selective_feedback/DEL_BASELINE.md` (+144 -0)
> - Wrote `./README.md` (+1 -1)
> - Ran `python -m unittest discover -s hardware_aware_selective_feedback/tests -t . -q`
> - Ran `git diff --stat; git status --short`
> - Ran `python -m hardware_aware_selective_feedback.run --include-del --device cuda --objective energy --calibration 1 --test 1 --tokens 8 --repeats 1 --observer-pairs 1 --output hardware_aware_selective_feedback/results/del-comparison/smoke.json`
>
> </details>
>
> All 64 package tests pass. The new DEL tests also pass in both installed environments: Transformers 4.50 and 5.18.
>
> The CUDA energy counter is available. I’ve started an FP32 smoke run, saving it under `results/del-comparison/` so your existing `latest.json` and `latest.md` stay intact.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `git diff -- hardware_aware_selective_feedback/run.py; Get-Content hardware_aware_selective_feedback/del_comparison.py`
> - Ran `Get-Content hardware_aware_selective_feedback/results/del-comparison/smoke.md; python -m hardware_aware_selective_feedback.run --include-del --device cuda --objective energy --calibration 2 --test 2 --tokens 32 --repeats 2 --observer-pairs 3 --output hardware_aware_selective_feedback/results/del-comparison/pilot.json`
>
> </details>
>
> The GPU smoke run passed: all 31 measured requests matched native greedy output, including DEL.
>
> The repeated pilot is now running with four held-out prompts, 32-token outputs, and two repeats. The smoke sample is too small for a performance conclusion; the pilot will compare adaptive, DEL, and fixed depth 12 on matched requests.
>
> <details><summary>Edited 3 files, ran 3 commands</summary>
>
> - Ran `Get-Content hardware_aware_selective_feedback/diagnose_mismatch.py -TotalCount 100; Get-Content hardware_aware_selective_feedback/tests/test_diagnose_mismatch.py; git diff --check; Get-Item hardware_aware_selective_feedback/results/del-comparison/smoke.json | Select-Object Length`
> - Ran `$r = Get-Content hardware_aware_selective_feedback/results/del-comparison/smoke.json -Raw | ConvertFrom-Json; $r.rows | Where-Object policy -eq del | ForEach-Object { [pscustomobject]@{Prompt=$_.prompt_id; Cycles=$_.del_cycles.Count; Depths=($_.del_cycles.depth -join ','); DraftLengths=(($_.del_cycles | ForEach-Object {$_.draft.Count}) -join ','); Energy=$_.energy_j} }; Get-Content hardware_aware_selective_feedback/diagnose_mismatch.py | Select-Object -Skip 130 -First 40`
> - Wrote `./hardware_aware_selective_feedback/DEL_BASELINE.md` (+3 -0)
> - Wrote `./hardware_aware_selective_feedback/diagnose_mismatch.py` (+7 -1)
> - Wrote `./hardware_aware_selective_feedback/tests/test_diagnose_mismatch.py` (+8 -0)
> - Ran `Get-Content hardware_aware_selective_feedback/diagnose_mismatch.py | Select-Object -Skip 123 -First 18; Get-Process python | Select-Object Id,CPU,WorkingSet64`
>
> </details>
>
> The repeated run is still in calibration and has reported no mismatches so far. I’m also recording an important comparison limit: DEL’s within-request decisions are included in its measured time and energy, while the existing controller’s request-level selection remains outside the generation meter. The report will label these as generation-cost comparisons, not end-to-end savings.
>
> <details><summary>Edited a file, ran 2 commands</summary>
>
> - Ran `$doc = Get-Content hardware_aware_selective_feedback/DEL_BASELINE.md -Raw; foreach ($m in [regex]::Matches($doc,'\]\(([^)]+)\)')) { $target=$m.Groups[1].Value; if ($target -notmatch '^https:' -and -not (Test-Path (Join-Path hardware_aware_selective_feedback $target))) { throw "Broken link: $target" } }; git diff --check; git status --short`
> - Wrote `./hardware_aware_selective_feedback/tests/test_del.py` (+27 -8)
> - Ran `nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,power.draw --format=csv,noheader; Get-Process -Id 35760 | Select-Object CPU,StartTime`
>
> </details>
>
> The pilot has completed the first of four held-out prompts with no reported mismatch. DEL is exercising its finer-grained choices: the smoke run selected depth 3 and varied draft lengths, rather than using the energy controller’s restricted depth set.
>
> Three of four held-out prompts are complete, still with no reported mismatch. Once the run finishes, I’ll check the paired energy results and whether fixed depth 12 performs as well as adaptation.
>
> System error
>
> You’ve hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/settings/usage to purchase more credits or try again at 11:07 PM.
</details>
