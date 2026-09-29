# Hardware-Aware Energy-Efficient LLM Inference through Model Compression and Adaptive Early Exit

A learning-focused, systematic experimental study of how model compression, hardware-aware quantization, and adaptive early-exit inference jointly affect LLM performance on consumer edge hardware (laptops).

## Research Question

How does the integration of model compression, hardware-aware quantization optimization, and adaptive early-exit inference affect the quality, latency, memory consumption, and energy efficiency of LLMs deployed on edge hardware?

## Approach

This project does not propose a new algorithm. It combines three existing ideas and studies their combined effect:

| Stage | Purpose |
|-------|---------|
| 1 | Compression: distillation, pruning, quantization |
| 2 | Hardware-aware configuration search with benchmark feedback |
| 3 | Adaptive early-exit inference (CALM-style) |

## Pipeline

```
Teacher LLM (Qwen3-1.7B)
        |  Knowledge Distillation
        v
Student LLM (Qwen3-0.6B)
        |  Structured Pruning
        v
Compressed LLM
        |  Quantization (FP16 / INT8 / INT4)
        v
Quantized LLM
        |  Hardware-aware configuration search + benchmark feedback
        v
Hardware-Optimized LLM
        |  CALM-style adaptive inference (confidence check per layer)
        v
Token generation  ->  Evaluation
```

## Models

- Teacher: Qwen3-1.7B
- Student: Qwen3-0.6B
- Reason: same tokenizer, low RAM, quick experiments, permissive license

## Evaluation Metrics

- Quality (task accuracy or perplexity)
- Latency
- Energy per token
- Peak memory
- Tokens per second

## Repository Structure

```
.
|-- README.md
|-- requirements.txt
|-- configs/            # seeds, dataset paths, run settings
|-- data/               # datasets (not committed if large)
|-- models/             # downloaded and generated models (git-ignored)
|-- stage1_compression/ # distillation, pruning, quantization
|-- stage2_hw_search/   # configuration search and benchmarking
|-- stage3_early_exit/  # CALM-style adaptive inference
|-- benchmarks/         # benchmark scripts and raw results per device
|-- results/            # tables and plots
`-- docs/               # notes, proposal, hardware specs
```

## Setup

1. Clone the repo
   ```
   git clone <repo-url>
   cd <repo-name>
   ```
2. Create a virtual environment
   ```
   python -m venv .venv
   # macOS/Linux: source .venv/bin/activate
   # Windows:     .venv\Scripts\activate
   ```
3. Install pinned dependencies
   ```
   pip install -r requirements.txt
   ```
4. Install llama.cpp (build from source or use a prebuilt release) for INT8/INT4 quantization and inference.


Energy measurement tools:

- Windows: HWiNFO, nvidia-smi, or CodeCarbon
- macOS: powermetrics, or CodeCarbon

## Experiment Flow

1. Baseline: run unmodified Qwen3-1.7B and Qwen3-0.6B on all four laptops
2. Stage 1 : compression only
3. Stage 2 : hardware-aware quantization search only
4. Stage 3 : early exit only
6. Compare each configuration against the baseline
