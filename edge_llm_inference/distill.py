"""
Stage 1a: Knowledge Distillation
Teacher: Qwen3-1.7B  ->  Student: Qwen3-0.6B
Dataset: tatsu-lab/alpaca (first NUM_EXAMPLES examples, for a quick first run)
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
IGNORE_INDEX = -100     # label value skipped by the loss

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(SEED)
print(f"Device: {device}")

#  1. Data 
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
    # Append EOS so the student learns where a response ends
    text = format_example(ex) + tokenizer.eos_token
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    # Mask padding by attention_mask, not pad_token_id, since pad may equal EOS
    enc["labels"] = [
        token_id if is_real_token else IGNORE_INDEX
        for token_id, is_real_token in zip(enc["input_ids"], enc["attention_mask"], strict=True)
    ]
    return enc


tokenized = raw.map(tokenize, remove_columns=raw.column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

#  2. Models 
print("Loading teacher...")
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_PATH, torch_dtype=torch.float32).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

print("Loading student...")
student = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, torch_dtype=torch.float32).to(device)
student.gradient_checkpointing_enable()
student.config.use_cache = False
student.train()

optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

#  3. Distillation loss 
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Logits at position t predict token t+1, so drop the last logit and the first label
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    target_labels = labels[:, 1:]
    valid_token_mask = target_labels != IGNORE_INDEX
    num_valid_tokens = valid_token_mask.sum().clamp(min=1)

    # Soft loss: per-token KL to the teacher's softened distribution, averaged over
    # real tokens only so it is on the same per-token scale as the hard loss
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    per_token_kl = F.kl_div(
        student_log_probs, teacher_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)
    soft_loss = (per_token_kl * valid_token_mask).sum() / num_valid_tokens * (temperature ** 2)

    # Hard loss: normal next-token prediction against the real labels
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        target_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    return alpha * soft_loss + (1 - alpha) * hard_loss


#  4. Training loop 
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

#  5. Save 
print(f"Saving distilled student to {OUTPUT_DIR}")
student.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")