"""
Finetune GPT-2 on ProsQA with standard Chain-of-Thought (CoT).
===============================================================

DDP version — launch with torchrun:

    # 3 GPUs (effective batch = 40 * 3 * 1 = 120, close to paper's 128)
    torchrun --nnodes 1 --nproc_per_node 3 finetune_cot.py \\
        --batch_size 40 --grad_accum 1

    # 2 GPUs (effective batch = 32 * 2 * 2 = 128, exact match)
    torchrun --nnodes 1 --nproc_per_node 2 finetune_cot.py \\
        --batch_size 32 --grad_accum 2

Training format (matches the paper's CoT baseline, run.py with cot=True):
    Input:  [Question]\\n [Step 1]\\n [Step 2]\\n ... ### [Answer] <eos>
    Labels: [-100 on question] [Step 1]\\n [Step 2]\\n ... ### [Answer] <eos>

Hyperparameters (Section 4.1, prosqa_coconut.yaml):
    lr=1e-4, effective_batch=128, wd=0.01, AdamW, 50 epochs, greedy eval
"""

import os
import sys
import json
import torch
import argparse
import itertools
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.config import BASE_GPT2, COT_GPT2, PROSQA_TRAIN, PROSQA_VAL


# ============================================================================
# DATA
# ============================================================================

def tokenize_dataset(path, tokenizer, max_size=None):
    """
    Tokenize ProsQA data identically to dataset.py get_dataset().

    Format per sample:
        question_tokenized: tokenizer.encode(question + "\\n", add_special_tokens=True)
        steps_tokenized:    [tokenizer.encode(step + "\\n", add_special_tokens=False) ...]
        answer_tokenized:   tokenizer.encode("### " + answer, ...) + [eos]
    """
    data = json.load(open(path))
    if max_size:
        data = data[:max_size]

    samples = []
    for idx, d in enumerate(data):
        q_tok = tokenizer.encode(d["question"] + "\n", add_special_tokens=True)
        s_tok = [
            tokenizer.encode(s + "\n", add_special_tokens=False) for s in d["steps"]
        ]
        a_tok = tokenizer.encode(
            "### " + d["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        samples.append({
            "question_tokenized": q_tok,
            "steps_tokenized": s_tok,
            "answer_tokenized": a_tok,
            "idx": idx,
        })

    return samples


def build_cot_training_sample(sample):
    """
    CoT training sample.
        tokens: [question] [step1] [step2] ... [answer]
        labels: [-100...]  [step1] [step2] ... [answer]
    """
    q = sample["question_tokenized"]
    steps_flat = list(itertools.chain.from_iterable(sample["steps_tokenized"]))
    a = sample["answer_tokenized"]

    input_ids = q + steps_flat + a
    labels = [-100] * len(q) + steps_flat + a

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def collate_fn(batch, pad_token_id, label_pad=-100):
    """Right-pad batch to max length."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        labels.append(b["labels"] + [label_pad] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ============================================================================
# EVALUATION (rank 0 only)
# ============================================================================

def evaluate_accuracy(model, tokenizer, val_path, device, max_new_tokens=128):
    """
    Greedy-decode CoT, extract answer after '###', compare to ground truth.
    Matches run.py lines 442-483.
    """
    data = json.load(open(val_path))

    # Unwrap DDP for generation
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for d in data:
            question_text = d["question"] + "\n"
            answer_gt = d["answer"].replace(",", "").strip()

            input_ids = tokenizer.encode(
                question_text, return_tensors="pt"
            ).to(device)

            output_ids = raw_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            text_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer_pred = text_output.split("#")[-1].replace(",", "").strip()

            correct += (answer_pred == answer_gt)
            total += 1

    return correct / total if total > 0 else 0.0


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    # --- Init distributed ---
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size: {world_size}")
        # effective_batch = batch_size * world_size * grad_accum
        eff = args.batch_size * world_size * args.grad_accum
        print(f"Effective batch size: {args.batch_size} * {world_size} * {args.grad_accum} = {eff}")

    # --- Model + tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(BASE_GPT2)
    tokenizer = AutoTokenizer.from_pretrained(BASE_GPT2)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # --- Data ---
    train_samples = tokenize_dataset(args.train_path, tokenizer)
    train_data = [build_cot_training_sample(s) for s in train_samples]

    sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=1,
        pin_memory=True,
    )

    if rank == 0:
        print(f"Training samples: {len(train_data)}")

    # --- Optimizer (AdamW, lr=1e-4, wd=0.01 — matches paper) ---
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0
    best_epoch = -1

    # --- Training loop ---
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum

        # --- Logging + eval (rank 0 only) ---
        avg_loss = total_loss / len(train_loader)

        # Reduce loss across ranks for accurate logging
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_global = loss_tensor.item() / world_size

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss_global:.4f}", end="")

        # Evaluate on rank 0
        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            if rank == 0:
                acc = evaluate_accuracy(model, tokenizer, args.val_path, device)
                print(f"  val_acc={acc:.4f}", end="")

                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    save_path = os.path.join(args.output_dir, "best_checkpoint.pt")
                    torch.save(model.module.state_dict(), save_path)
                    print(f"  [saved best]", end="")

        if rank == 0:
            print()

        # Periodic checkpoint (rank 0 saves, all ranks barrier)
        if (epoch + 1) % 10 == 0:
            if rank == 0:
                save_path = os.path.join(
                    args.output_dir, f"checkpoint_{epoch + 1}.pt"
                )
                torch.save(model.module.state_dict(), save_path)
            dist.barrier()

        sys.stdout.flush()

    # --- Final save ---
    if rank == 0:
        save_path = os.path.join(args.output_dir, "final_checkpoint.pt")
        torch.save(model.module.state_dict(), save_path)
        print(f"\nDone. Best val accuracy: {best_acc:.4f} at epoch {best_epoch}")
        print(f"Checkpoints in {args.output_dir}/")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=str(PROSQA_TRAIN))
    parser.add_argument("--val_path", type=str, default=str(PROSQA_VAL))
    parser.add_argument("--output_dir", type=str, default=str(COT_GPT2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=1)
    args = parser.parse_args()
    main(args)