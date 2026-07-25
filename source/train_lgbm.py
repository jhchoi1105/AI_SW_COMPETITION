# -*- coding: utf-8 -*-
"""[1단계] LightGBM 파이프라인 학습 (텍스트 모델 + 스태킹 LGBM).

산출물 (./work/lgbm/):
  - text_model.pkl    : TF-IDF(1~2gram) + LogisticRegression, 전체 70,000건 학습
  - lgbm_final.txt    : 스태킹 LightGBM (텍스트 OOF확률 14 + 구조 49 + 표적 11 + ID 13)
  - lgbm_classes.json : 클래스 순서

피처 함수(build_text/build_struct/extra_features/id_features)는 최종 추론 코드
final_inference/script.py에서 그대로 import — 학습·추론 피처 정의가 단일 원천으로 일치함.

실행:  python train_lgbm.py   (이 폴더에서, ./data/train.jsonl 필요)
"""
import csv
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_inference"))
from script import build_text, build_struct, extra_features, id_features  # noqa: E402

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

OUT = "./work/lgbm"
OOF_CACHE = "./work/_oof_full_cache.npy"

# LGBM categorical feature 인덱스.
# 입력 벡터 = [텍스트 OOF확률 14] + [구조 49] + [표적 11] + [ID 13] 순서이며,
# 구조 블록 내 카테고리 피처는 7~13번(CI상태, user_tier, language_pref,
# 주 언어, 직전 행동 1·2·3) → OOF 오프셋 14를 더해 21~27.
LGBM_CAT_IDX = [21, 22, 23, 24, 25, 26, 27]


def make_text_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80_000,
                                  sublinear_tf=True, lowercase=True)),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=2.0)),
    ])


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = [json.loads(l) for l in open("./data/train.jsonl", encoding="utf-8") if l.strip()]
    labels = {r["id"]: r["action"]
              for r in csv.DictReader(open("./data/train_labels.csv", encoding="utf-8"))}
    y = np.array([labels[s["id"]] for s in samples])
    groups = np.array([s["id"].split("-step_")[0] for s in samples])
    texts = [build_text(s) for s in samples]

    # 1) 추론용 텍스트 모델: 전체 데이터로 학습
    if not os.path.exists(f"{OUT}/text_model.pkl"):
        t0 = time.time()
        full_pipe = make_text_pipeline()
        full_pipe.fit(texts, y)
        joblib.dump(full_pipe, f"{OUT}/text_model.pkl")
        print(f"text_model.pkl 저장 ({time.time()-t0:.0f}s)", flush=True)

    # 2) 스태킹용 텍스트 OOF 확률 (세션 그룹 분리 GroupKFold 5)
    if os.path.exists(OOF_CACHE):
        oof = np.load(OOF_CACHE)
        print("OOF 캐시 로드", flush=True)
    else:
        oof = np.zeros((len(samples), 14))
        for k, (tr, va) in enumerate(GroupKFold(5).split(texts, y, groups)):
            t0 = time.time()
            pipe = make_text_pipeline()
            pipe.fit([texts[i] for i in tr], y[tr])
            oof[va] = pipe.predict_proba([texts[i] for i in va])
            print(f" OOF fold {k}: {time.time()-t0:.0f}s", flush=True)
        np.save(OOF_CACHE, oof)

    # 3) 구조/표적/ID 피처 결합 → LightGBM 학습
    struct = np.asarray([build_struct(s) for s in samples])
    extra = np.asarray([extra_features(s) for s in samples])
    ids = np.asarray([id_features(s) for s in samples])
    x = np.hstack([oof, struct, extra, ids])
    print(f"X={x.shape}", flush=True)

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=14, n_estimators=500,
        learning_rate=0.03, num_leaves=63, min_child_samples=50,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    model.fit(x, y, sample_weight=compute_sample_weight("balanced", y),
              categorical_feature=LGBM_CAT_IDX)
    print(f"LGBM 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

    model.booster_.save_model(f"{OUT}/lgbm_final.txt")
    with open(f"{OUT}/lgbm_classes.json", "w", encoding="utf-8") as f:
        json.dump([str(c) for c in model.classes_], f)
    print(f"저장 완료: {OUT}", flush=True)


if __name__ == "__main__":
    main()
