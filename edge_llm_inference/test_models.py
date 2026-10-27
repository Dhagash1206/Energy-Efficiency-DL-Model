
import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "Qwen3-0.6B (student)": "models/qwen3-0.6b",
    "Qwen3-1.7B (teacher)": "models/qwen3-1.7b",
}
PROMPT = "Explain what an LLM is in one sentence."
MAX_NEW_TOKENS = 50
SEED = 42

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}\n")


def run(name, path):
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to(device)
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Warm-up run so timing is fair
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)

    if device == "mps":
        torch.mps.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - start

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    mem_gb = torch.mps.current_allocated_memory() / 1e9 if device == "mps" else float("nan")

    print(f"=== {name} ===")
    print(f"Reply: {reply.strip()}")
    print(f"Tokens generated: {new_tokens}")
    print(f"Time: {elapsed:.2f} s")
    print(f"Tokens/second: {new_tokens / elapsed:.1f}")
    print(f"Latency: {1000 * elapsed / new_tokens:.1f} ms/token")
    print(f"GPU memory: {mem_gb:.2f} GB\n")

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


for name, path in MODELS.items():
    run(name, path)

print("Both models ran successfully.")