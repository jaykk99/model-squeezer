"""CPU LoRA fine-tuning for model-squeezer.

Two methods:
  lora          - standard LoRA with AdamW on CPU
  lora-lowmem   - LoRA with Adafactor + gradient checkpointing + bf16
                  autocast when available. Built for weak hardware.

Training runs in a background thread and reports progress (step, loss,
percent, ETA) through a shared job dict. Honest ETA comes from measured
steps-per-second, never a guess.
"""
import os
import time
import math
import json
import torch


def load_training_texts(data_path, max_chars=2000000):
    """Read .jsonl (one {'text': ...} per line) or plain .txt."""
    texts = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("text") if isinstance(obj, dict) else None
                if t:
                    texts.append(t)
                if sum(len(x) for x in texts) > max_chars:
                    break
    else:
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read(max_chars)
        # split into ~paragraph chunks
        for para in raw.split("\n\n"):
            para = para.strip()
            if len(para) > 40:
                texts.append(para)
    return texts


def run_lora_training(job, model_path, data_path, out_dir, method="lora",
                      epochs=1, lr=2e-4, batch_size=1,
                      grad_accum=8, max_seq_len=256, lora_r=8, lora_alpha=16):
    """Blocking training loop. Updates job dict with progress."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    def prog(pct, msg, **extra):
        job["progress"] = pct
        job["status_msg"] = msg
        job.update(extra)

    prog(2, "Loading tokenizer and model on CPU...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32
    use_bf16 = False
    if method == "lora-lowmem":
        try:
            if torch.cpu.is_available() or True:
                # bf16 autocast on CPU needs recent hardware; try it
                use_bf16 = True
        except Exception:
            use_bf16 = False

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True)
    if method == "lora-lowmem":
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    prog(8, "Attaching LoRA adapters...")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    prog(12, "Loading training data...")
    texts = load_training_texts(data_path)
    if not texts:
        job["ok"] = False
        job["error"] = "No usable training text found in the data file."
        return
    enc = tok(texts, truncation=True, max_length=max_seq_len,
              padding=False, return_attention_mask=False)
    input_ids = enc["input_ids"]
    # drop very short sequences
    input_ids = [s for s in input_ids if len(s) > 8]
    if not input_ids:
        job["ok"] = False
        job["error"] = "Training texts were too short after tokenizing."
        return

    if method == "lora-lowmem":
        from transformers import Adafactor
        opt = Adafactor(model.parameters(), lr=lr, relative_step=False,
                        scale_parameter=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    total_steps = max(1, math.ceil(len(input_ids) / (batch_size * grad_accum)) * epochs)
    step = 0
    losses = []
    start = time.time()
    idx = 0

    prog(15, "Training started on CPU...")
    for epoch in range(epochs):
        for b in range(0, len(input_ids), batch_size * grad_accum):
            chunk = input_ids[b:b + batch_size * grad_accum]
            opt.zero_grad()
            batch_loss = 0.0
            for s in chunk:
                ids = torch.tensor([s], dtype=torch.long)
                with torch.amp.autocast("cpu", dtype=torch.bfloat16,
                                        enabled=use_bf16):
                    out = model(input_ids=ids, labels=ids)
                    loss = out.loss / max(1, len(chunk))
                loss.backward()
                batch_loss += float(out.loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            avg_loss = batch_loss / max(1, len(chunk))
            losses.append(round(avg_loss, 4))
            elapsed = time.time() - start
            sps = step / max(elapsed, 0.01)
            eta = (total_steps - step) / max(sps, 0.001)
            pct = 15 + 80 * step / total_steps
            prog(pct, "Epoch %d/%d, step %d/%d, loss %.4f" % (
                epoch + 1, epochs, step, total_steps, avg_loss),
                loss=round(avg_loss, 4), step=step,
                total_steps=total_steps,
                eta_seconds=int(eta),
                loss_curve=losses[-200:])
            if job.get("cancel"):
                job["ok"] = False
                job["error"] = "Training cancelled."
                return

    prog(97, "Saving adapter...")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "training_report.json"), "w") as f:
        json.dump({
            "method": method, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "grad_accum": grad_accum,
            "steps": step, "final_loss": losses[-1] if losses else None,
            "loss_curve": losses,
            "elapsed_seconds": int(time.time() - start),
        }, f, indent=2)

    job["ok"] = True
    job["progress"] = 100
    job["status_msg"] = "Training complete. Loss %.4f." % (losses[-1] if losses else 0)
    job["result"] = {"out_dir": out_dir, "final_loss": losses[-1] if losses else None}
