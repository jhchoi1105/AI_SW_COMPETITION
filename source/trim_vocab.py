# -*- coding: utf-8 -*-
"""xlm-r 임베딩 어휘 축소 — 용량 1GB 제한 대응.

원리: 25만 어휘 임베딩(fp16 384MB)이 모델의 70%를 차지하는데, train 직렬화
텍스트에서 실제 등장하는 토큰은 일부뿐. 등장 토큰만 남긴 임베딩으로 교체하고,
추론 시 [전체 id → 축소 id] 재매핑 배열을 적용한다 (미등장 토큰 → <unk>).
토크나이저는 원본 그대로 사용하므로 토크나이저 수술이 필요 없다.

실행:  python scripts/trim_vocab.py --src ./strategy_c_work/xlmr_s7e6_100 --dst ./strategy_c_work/xlmr_100_trimmed
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
import serialize_full  # noqa: E402

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--ser", default="v1", choices=["v1", "v2", "v3"])
    args = ap.parse_args()
    build_text_full = {
        "v1": serialize_full.build_text_full,
        "v2": serialize_full.build_text_full_v2,
        "v3": serialize_full.build_text_full_v3,
    }[args.ser]

    tok = AutoTokenizer.from_pretrained(args.src)
    samples = [json.loads(l) for l in open("./data/train.jsonl", encoding="utf-8") if l.strip()]
    texts = [build_text_full(s) for s in samples]
    print("train 토큰 수집...")
    used = set()
    enc = tok(texts, truncation=True, max_length=args.max_len)
    for ids in enc["input_ids"]:
        used.update(ids)
    # 특수 토큰 + unk는 반드시 유지
    for t in [tok.pad_token_id, tok.unk_token_id, tok.cls_token_id, tok.sep_token_id,
              tok.bos_token_id, tok.eos_token_id, tok.mask_token_id]:
        if t is not None:
            used.add(t)
    keep = sorted(used)
    print(f"유지 토큰: {len(keep):,} / {tok.vocab_size:,} ({len(keep)/tok.vocab_size:.1%})")

    model = AutoModelForSequenceClassification.from_pretrained(args.src)
    emb = model.get_input_embeddings().weight.data  # (V, H)
    unk = tok.unk_token_id
    # 재매핑: 전체 id → 축소 행 (미등장 → unk의 축소 행)
    old2new = np.full(emb.shape[0], -1, dtype=np.int64)
    for new, old in enumerate(keep):
        old2new[old] = new
    unk_new = int(old2new[unk])
    old2new[old2new == -1] = unk_new

    new_emb = emb[torch.tensor(keep)].clone()
    model.get_input_embeddings().weight.data = new_emb
    model.config.vocab_size = len(keep)
    # position embeddings 등은 그대로. resize 검증:
    assert model.get_input_embeddings().weight.shape[0] == len(keep)

    os.makedirs(args.dst, exist_ok=True)
    model.half().save_pretrained(args.dst)
    tok.save_pretrained(args.dst)
    np.save(os.path.join(args.dst, "id_remap.npy"), old2new)
    for f in ("classes.json",):
        srcf = os.path.join(args.src, f)
        if os.path.exists(srcf):
            import shutil
            shutil.copy(srcf, os.path.join(args.dst, f))
    total = sum(os.path.getsize(os.path.join(args.dst, f)) for f in os.listdir(args.dst))
    print(f"저장: {args.dst} ({total/1e6:.0f} MB)")

    # ---- 파리티 검증: 원본 vs 축소 (val 500건, GPU) ----
    print("파리티 검증...")
    import csv
    from sklearn.model_selection import GroupShuffleSplit
    labels = {r["id"]: r["action"]
              for r in csv.DictReader(open("./data/train_labels.csv", encoding="utf-8"))}
    y = np.array([labels[s["id"]] for s in samples])
    groups = np.array([s["id"].rsplit("-step_", 1)[0] for s in samples])
    _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(texts, y, groups))
    sub = va[:500]
    sub_texts = [texts[i] for i in sub]

    orig = AutoModelForSequenceClassification.from_pretrained(args.src, torch_dtype=torch.float16).cuda().eval()
    trim = AutoModelForSequenceClassification.from_pretrained(args.dst, torch_dtype=torch.float16).cuda().eval()

    def predict(mdl, remap=None):
        enc2 = tok(sub_texts, truncation=True, max_length=args.max_len)
        preds = []
        with torch.no_grad():
            for i in range(0, len(sub_texts), 32):
                batch = enc2["input_ids"][i:i+32]
                ml = max(len(b) for b in batch)
                ids = np.full((len(batch), ml), tok.pad_token_id, dtype=np.int64)
                mask = np.zeros((len(batch), ml), dtype=np.int64)
                for r, bb in enumerate(batch):
                    ids[r, :len(bb)] = bb
                    mask[r, :len(bb)] = 1
                if remap is not None:
                    ids = remap[ids]
                logits = mdl(input_ids=torch.from_numpy(ids).cuda(),
                             attention_mask=torch.from_numpy(mask).cuda()).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
        return np.array(preds)

    p_orig = predict(orig)
    p_trim = predict(trim, old2new)
    print(f"예측 일치율: {(p_orig == p_trim).mean():.4f} (500건)")


if __name__ == "__main__":
    main()
