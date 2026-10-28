"""Fixed early-exit block indices for the study models.

Blocks are counted from 1. The last exit is always the final block.
"""

EXIT_BLOCKS = {
    "HuggingFaceTB/SmolLM2-135M": (8, 15, 23, 30),
    "Qwen/Qwen3-0.6B": (7, 14, 21, 28),
    "facebook/layerskip-llama3.2-1B": (4, 8, 12, 16),
}


def exit_blocks(model_name: str, num_hidden_layers: int) -> tuple[int, ...]:
    """Return 1-based exit blocks, checking them against the loaded config."""
    if model_name not in EXIT_BLOCKS:
        known = ", ".join(sorted(EXIT_BLOCKS))
        raise KeyError(f"No exit map for {model_name}. Known models: {known}")
    blocks = EXIT_BLOCKS[model_name]
    if blocks[-1] != num_hidden_layers:
        raise ValueError(
            f"{model_name} exit map ends at block {blocks[-1]}, "
            f"but the checkpoint has {num_hidden_layers} blocks"
        )
    if any(block < 1 or block > num_hidden_layers for block in blocks):
        raise ValueError(f"Exit blocks {blocks} do not fit {num_hidden_layers} layers")
    return blocks
