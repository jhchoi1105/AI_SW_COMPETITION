# -*- coding: utf-8 -*-
"""[전략 C 강화판] 전체 history 직렬화 + 파라미터화 파인튜닝.

개선점 (train_strategy_c.py 대비):
- 입력: build_text_full (사용자 발화·행동(인자)·결과 전체, 최근순 → 잘려도 옛 턴부터)
- 길이 버킷팅 배치 (패딩 낭비 제거로 학습 속도 확보)
- 모델/에폭/길이/배치 파라미터화 → xlm-r vs mdeberta 비교 실험
- 최고 에폭의 검증 확률(np) 저장 → 이후 앙상블/threshold 실험 재료

실행 예:
  python scripts/train_strategy_c2.py --model xlm-roberta-base --epochs 4 --batch 24 --out ./strategy_c_work/xlmr_full
  python scripts/train_strategy_c2.py --model microsoft/mdeberta-v3-base --epochs 4 --batch 16 --out ./strategy_c_work/mdeberta_full
"""
import argparse
import csv
import gc
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
import serialize_full  # noqa: E402

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
CLASSES = sorted(ALL_CLASSES)
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


def bucket_batches(lengths, batch_size, shuffle=True, seed=0):
    """길이순 정렬 후 배치 단위로 묶고 배치 순서만 셔플 → 패딩 최소화."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def make_batch(enc, idx, pad_id, labels=None):
    maxlen = max(len(enc["input_ids"][i]) for i in idx)
    ids = torch.full((len(idx), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros(len(idx), maxlen, dtype=torch.long)
    for r, i in enumerate(idx):
        n = len(enc["input_ids"][i])
        ids[r, :n] = torch.tensor(enc["input_ids"][i])
        mask[r, :n] = 1
    y = torch.tensor([labels[i] for i in idx], dtype=torch.long) if labels is not None else None
    return ids, mask, y


def evaluate(model, enc, y, batches, pad_id, device, amp_dtype=None):
    import contextlib
    model.eval()
    n = len(y)
    probs = np.zeros((n, 14), dtype=np.float32)
    with torch.no_grad():
        for idx in batches:
            ids, mask, _ = make_batch(enc, idx, pad_id)
            ctx = (contextlib.nullcontext() if amp_dtype is None
                   else torch.autocast("cuda", dtype=amp_dtype))
            with ctx:
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            p = torch.softmax(logits.float(), -1).cpu().numpy()
            for r, i in enumerate(idx):
                probs[i] = p[r]
    preds = probs.argmax(-1)
    return f1_score(y, preds, average="macro"), probs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--epoch-offset", type=int, default=0,
                    help="재개 학습 시 로그·배치 셔플에 더할 완료 epoch 수")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true",
                    help="검증 분할 없이 전체 70k로 학습 (최종 제출용, 에폭 수 고정)")
    ap.add_argument("--seed", type=int, default=42, help="torch 시드 (분류 헤드 초기화·셔플)")
    ap.add_argument("--ser", default="v1", choices=["v1", "v2", "v3"],
                    help="직렬화 버전 (v2=ID토큰, v3=ID+구조신호 토큰)")
    ap.add_argument("--fp32", action="store_true",
                    help="autocast 끄고 fp32로 학습 (mDeBERTa 등 bf16 NaN 회피)")
    ap.add_argument("--fp16", action="store_true",
                    help="T4 등 bf16 미지원 GPU에서 fp16 + loss scaling 사용")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="활성화 메모리를 줄여 큰 모델을 작은 GPU에서 학습")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Hugging Face custom model code 허용 (Alibaba GTE 등에 필요)")
    args = ap.parse_args()

    import contextlib

    if args.fp32 and args.fp16:
        ap.error("--fp32와 --fp16은 동시에 사용할 수 없습니다")
    amp_dtype = None if args.fp32 else (torch.float16 if args.fp16 else torch.bfloat16)

    def amp_ctx():
        return (contextlib.nullcontext() if amp_dtype is None
                else torch.autocast("cuda", dtype=amp_dtype))
    build_text_full = {
        "v1": serialize_full.build_text_full,
        "v2": serialize_full.build_text_full_v2,
        "v3": serialize_full.build_text_full_v3,
    }[args.ser]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    assert torch.cuda.is_available()
    if args.fp32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"== {args.model} | ep={args.epochs} len={args.max_len} bs={args.batch} lr={args.lr} ==", flush=True)

    samples = [json.loads(l) for l in open("./data/train.jsonl", encoding="utf-8") if l.strip()]
    labels_map = {r["id"]: r["action"]
                  for r in csv.DictReader(open("./data/train_labels.csv", encoding="utf-8"))}
    texts = [build_text_full(s) for s in samples]
    y = np.array([CLS_IDX[labels_map[s["id"]]] for s in samples])
    groups = np.array([s["id"].split("-step_")[0] for s in samples])

    if args.full:
        tr = np.arange(len(texts))
        va = np.arange(0, len(texts), 50)  # 로깅용 소규모 (학습에 포함됨 — 점수는 참고만)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, va = next(gss.split(texts, y, groups))

    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    pad_id = tok.pad_token_id
    enc_tr = tok([texts[i] for i in tr], truncation=True, max_length=args.max_len)
    enc_va = tok([texts[i] for i in va], truncation=True, max_length=args.max_len)
    # make_batch가 input_ids만 사용하고 mask를 직접 만들기 때문에 tokenizer가 만든
    # attention_mask/token_type_ids를 보관할 필요가 없다. Python 정수 리스트라 전체
    # 70k에서는 수 GB를 차지해 epoch 저장 뒤 페이징을 유발할 수 있다.
    enc_tr = {"input_ids": enc_tr["input_ids"]}
    enc_va = {"input_ids": enc_va["input_ids"]}
    y_tr, y_va = y[tr], y[va]
    len_tr = [len(x) for x in enc_tr["input_ids"]]
    len_va = [len(x) for x in enc_va["input_ids"]]
    va_batches = bucket_batches(len_va, 96, shuffle=False)
    del samples, texts, labels_map, groups
    gc.collect()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=14,
        trust_remote_code=args.trust_remote_code,
    )
    # 일부 로컬 체크포인트는 config와 무관하게 fp16 tensor를 포함한다.
    # mDeBERTa의 fp16 학습은 NaN이 발생했으므로 --fp32에서 명시적으로 승격한다.
    if args.fp32:
        model = model.float()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = model.to(device)

    counts = np.bincount(y_tr, minlength=14)
    w = torch.tensor(len(y_tr) / (14.0 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        foreach=False if args.fp32 else None,
    )
    steps_per_ep = (len(tr) + args.batch - 1) // args.batch
    total = steps_per_ep * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total * args.warmup_ratio), total,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best_f1, best_state, best_probs = -1.0, None, None
    for ep in range(args.epochs):
        actual_ep = args.epoch_offset + ep
        model.train()
        batches = bucket_batches(
            len_tr, args.batch, shuffle=True, seed=args.seed + actual_ep,
        )
        t0, running, seen = time.time(), 0.0, 0
        for step, idx in enumerate(batches):
            ids, mask, yb = make_batch(enc_tr, idx, pad_id, y_tr)
            with amp_ctx():
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = loss_fn(logits.float(), yb.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={actual_ep}, step={step}, value={loss.item()}"
                )
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            seen += 1
            if (step + 1) % 400 == 0:
                print(f" ep{actual_ep} {step+1}/{len(batches)} loss={running/seen:.4f} ({time.time()-t0:.0f}s)", flush=True)
                running, seen = 0.0, 0
        f1, probs, _ = evaluate(model, enc_va, y_va, va_batches, pad_id, device, amp_dtype=amp_dtype)
        tag = "train-subset F1(참고용)" if args.full else "val Macro-F1"
        print(f"[epoch {actual_ep}] {tag} = {f1:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if args.full or f1 > best_f1:
            best_f1, best_probs = f1, probs
            # full 모드는 매 epoch 최신본을 채택하므로 CPU에 1GB+ state clone을
            # 유지할 필요가 없다. 검증 모드에서만 최고 epoch 복원을 위해 보관한다.
            if not args.full:
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            # 프로세스가 도중에 죽어도 최고 에폭이 남도록 즉시 저장
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            with open(os.path.join(args.out, "classes.json"), "w", encoding="utf-8") as f:
                json.dump(CLASSES, f)
            np.save(os.path.join(args.out, "val_probs.npy"), probs)
            np.save(os.path.join(args.out, "val_idx.npy"), va)
            print(f" 중간 저장 완료 (ep{actual_ep}, F1={f1:.4f})", flush=True)

    print(f"\nbest val Macro-F1 = {best_f1:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, preds = evaluate(model, enc_va, y_va, va_batches, pad_id, device, amp_dtype=amp_dtype)
    print(classification_report(y_va, preds, target_names=CLASSES, digits=3, zero_division=0))

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(CLASSES, f)
    np.save(os.path.join(args.out, "val_probs.npy"), best_probs)
    np.save(os.path.join(args.out, "val_idx.npy"), va)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
