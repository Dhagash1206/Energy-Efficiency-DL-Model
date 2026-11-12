# Coding-agent handoff: publication build

Use this with Sol, Gemini, or another coding agent that can inspect this repository.
The prompts are model-independent; they do not depend on a particular model version.
The [research and implementation plan](publication-build-plan.md) explains the
evidence, proposed design, primary sources, and acceptance gates. The existing
[publication protocol](publication.md) documents implemented commands.

## Start here

Give the agent repository access and paste the master prompt below. Start with
**P0**, then **P1**. Do not request all eight phases in one implementation pass.
New filenames and interfaces in the plan are proposals until the agent implements
and verifies them; do not assume they are available commands.

```text
You are implementing the publication build plan in this repository.

Read applicable AGENTS.md instructions, docs/publication-build-plan.md,
docs/publication.md, README.md, and the relevant source/tests before editing.
Inspect git status and preserve existing uncommitted work and saved results.

Implement PHASE P0 only in this session. If P0 is already complete, verify its
evidence and report the next phase rather than repeating work.

The scientific objective is to establish credible task quality, then test whether
adaptive self-speculative decoding saves total GPU energy against a strong fixed
configuration selected on development data. Task accuracy, token agreement, and
software-test success are different metrics. Never substitute one for another.

The current FP16 pilot has 57/57 native-greedy matches, but every policy scored
0/2 GSM8K and 0/2 HumanEval. All held-out completions hit the 256-token limit.
Fixed depth 4 consumed less energy than adaptive in that pilot. These are small
development observations, not publication claims or proof that FP16 is defective.

Work one phase at a time. Preserve raw reports and use fresh output paths.
Do not change scorers, prompts, or stop rules based on final-test answers.
Do not train on inspected benchmark answers or repair generated code using tests.
Do not claim planned experiments ran. Keep historical and new protocols separate.

Implement the smallest coherent change satisfying this phase's acceptance gate.
Add meaningful tests for changed behavior, run relevant checks, and record exact
commands and outcomes. For model runs, measure memory and request cost on a tiny
development batch before proposing a larger schedule. Run one GPU job at a time.
Do not download larger checkpoints, start training, or run a full benchmark as
part of this initial phase. Keep the current experiment environment reproducible.

Create or update docs/build-progress.md with phase status, changed files, checks,
artifact paths, unresolved findings, and the next bounded task. It is a progress
ledger, not a place to mark plans complete without evidence.

Finish with: what changed; validation and limitations; the phase gate status;
and the exact next task. Do not commit or push unless separately requested.
```

## Phase-specific assignments

After the first session, replace the phase instruction in the master prompt with
one assignment below. Carry the same evidence and result-preservation rules forward.

| Phase | Assignment | Evidence needed before advancing |
|---|---|---|
| P0 | Add reproducibility manifests, source/dirty-tree attribution, and an explicit development/test exposure registry. Separate per-dataset IDs/counts from the current coupled loader. | Stable prompt IDs and hashes; complete runtime/model/data provenance; repeatable split selection; unique result paths |
| P1 | Implement pinned benchmark protocols, independent reference generation/scorer checks, and useful failure categories. | Per-item prompt/scorer parity on fixtures and saved samples; infrastructure errors separate from model failures |
| P2 | Implement one committed-output stopping contract for ordinary, speculative, and DEL paths. | EOS, multi-token boundaries, rejected draft stops, and accepted-block stops tested; tokens/text/cache consistent |
| P3 | Run a small full-depth development study using the verified protocol; expand only after inspecting failure causes. | Capability, truncation, repetition, precision, and checkpoint effects separated; explicit model-quality decision |
| P4 | Select the best fixed action on development data and freeze it; add a competitive upstream ordinary baseline and matched prior-work controls. | Same checkpoint/protocol/action information; no test-selected winner; tuning cost recorded |
| P5 | Add complete request cost accounting, cumulative online traces, fair setup-cost allocation, and uncertainty-aware break-even analysis. | Controller/probe/calibration overhead included or explicitly bounded; sensor units/scope verified; synthetic ledgers tested |
| P6 | Compare simple frozen, online no-probe, and online paid-probe selectors with matched information and acquisition budgets. | Separate executed/observer feedback estimators; no hidden future information; feedback gain survives its measured overhead |
| P7 | Freeze the final protocol, run the justified evaluation schedule, and generate the reproducibility/paper evidence bundle. | Prespecified comparisons and sample-size rationale; paired or stream-level uncertainty; all exclusions and negative results visible |

P3 is a decision gate, not a mandate to train. If the full-depth checkpoint is
unsuitable, use the plan's model-improvement branch. If P6 cannot beat the tuned
fixed baseline, report that result and retain fixed decoding as the practical choice.

## Prompt for the first evaluation implementation

Use after P0 is complete:

```text
Implement P1 from docs/publication-build-plan.md only.
Inspect run.py:load_prompts, engine.py:native_greedy, quality.py, and their tests.
Pin the reference harness revision and resolve its actual task configuration.
Do not mix the standard five-shot GSM8K task with a separate eight-shot CoT task.
For HumanEval, preserve one raw greedy completion per task and the declared
completion/stop protocol. Compare identical saved completions across evaluators.

Build fixtures for correct/incorrect/absent/multiple math answers and correct,
incorrect, syntax-error, timeout, and infrastructure-error code evaluations.
Keep generated code inside the existing isolated execution boundary.
Make provenance and per-item differences inspectable. Gold answers/tests must
not enter generation prompts, selector inputs, or output-repair logic.

Demonstrate scorer and prompt parity first. If model generation is needed, use
only a tiny registered development batch. Report any stopping differences as
P2 work rather than silently changing multiple protocols to get higher scores.
```

## Prompt for model-quality work

Use only after P1/P2 evidence is available:

```text
Execute P3's development diagnostics with the verified evaluation protocol.
First evaluate the current checkpoint at full depth. Diagnose wrong reasoning,
wrong function behavior, repetition, formatting, and truncation separately.
Compare precision within the same software stack. Record actual prompt lengths,
finish reasons, raw outputs, task scores, memory peaks, and measured runtime.

Recommend whether to retain the checkpoint, use a stronger existing exit-trained
model on suitable hardware, or conduct exit-aware adaptation. Separate a more
capable full-depth instruction model from a useful self-speculative target.
Do not assume ordinary LoRA preserves early-exit acceptance or that 4-GB VRAM
supports training merely because inference weights fit.

If training is warranted, prepare a bounded experiment specification before
execution: licensed data, contamination checks, teacher provenance if applicable,
memory/token budget, original/SFT/exit-aware controls, and evaluation gates.
No invented target score or speedup. Preserve unsuccessful conditions.
```

## Validation and progress contract

Existing repository checks, run from its root with the intended environment:

```powershell
python -B -m unittest discover -s hardware_aware_selective_feedback/tests -t . -v
python -B -m unittest prototype.tests.test_prototype -v
python -m ruff format --check .
python -m ruff check .
```

Use focused tests during implementation and the relevant full checks at a phase
boundary. Hardware tests require separately recorded evidence. A unit-test pass
does not certify GPU decoding equivalence or benchmark accuracy.

Each progress entry should contain:

1. Phase and status: proposed, in progress, passed, or unresolved.
2. Source revision plus working-tree attribution and relevant environment.
3. Changed behavior and files, including protocol/schema version changes.
4. Exact validation commands and observed outcomes, including failures/skips.
5. New artifact paths, experiment settings, and task exposure updates.
6. Acceptance criteria met and remaining gaps.
7. The next smallest task, with its inputs and expected outputs.

For a fresh agent session, paste:

```text
Read docs/agent-handoff.md, docs/publication-build-plan.md, and
docs/build-progress.md. Verify the latest phase evidence against the current
files and results. Continue only the next bounded task recorded in the ledger.
If the ledger is absent, start with P0. If evidence contradicts a completion
claim, document the discrepancy and fix that gate before advancing.
```
