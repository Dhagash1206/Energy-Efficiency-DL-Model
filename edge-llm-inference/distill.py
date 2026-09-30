"""
Stage 1a: Knowledge Distillation
Teacher: Qwen3-1.7B  ->  Student: Qwen3-0.6B
Dataset: tatsu-lab/alpaca (first 1000 examples, for a quick first run)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_PATH = "models/qwen3-1.7b"
STUDENT_PATH = "models/qwen3-0.6b"
OUTPUT_DIR = "models/qwen3-0.6b-distilled"

NUM_EXAMPLES = 200
MAX_LENGTH = 128
BATCH_SIZE = 1
EPOCHS = 1
LR = 5e-6
TEMPERATURE = 2.0        # softens teacher probabilities
ALPHA = 0.5              # weight between distillation loss and normal LM loss
SEED = 42

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(SEED)
print(f"Device: {device}")

# ---------- 1. Data ----------
print("Loading dataset...")
raw = load_dataset("tatsu-lab/alpaca", split=f"train[:{NUM_EXAMPLES}]")

tokenizer = AutoTokenizer.from_pretrained(STUDENT_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_example(ex):
    if ex["input"]:
        prompt = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nResponse: {ex['output']}"
    else:
        prompt = f"Instruction: {ex['instruction']}\nResponse: {ex['output']}"
    return prompt


def tokenize(ex):
    text = format_example(ex)
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc


tokenized = raw.map(tokenize, remove_columns=raw.column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

# ---------- 2. Models ----------
print("Loading teacher...")
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_PATH, dtype=torch.float32).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

print("Loading student...")
student = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, dtype=torch.float32).to(device)
student.gradient_checkpointing_enable()
student.config.use_cache = False
student.train()

optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

# ---------- 3. Distillation loss ----------
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Soft loss: student matches teacher's softened distribution (KL divergence)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    # Hard loss: normal next-token prediction against the real labels
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )
    return alpha * soft_loss + (1 - alpha) * hard_loss


# ---------- 4. Training loop ----------
print("Starting training...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_out.logits

        student_out = student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_out.logits

        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del teacher_out, teacher_logits, student_out, student_logits
        if device == "mps":
            torch.mps.empty_cache()

        total_loss += loss.item()
        if step % 20 == 0:
            print(f"Epoch {epoch+1} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} finished. Avg loss: {total_loss / len(loader):.4f}")

# ---------- 5. Save ----------
print(f"Saving distilled student to {OUTPUT_DIR}")
student.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")