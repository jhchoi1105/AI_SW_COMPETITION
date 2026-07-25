# -*- coding: utf-8 -*-
"""전체 학습 파이프라인 실행 — Private Score(0.79677, g4_c15) 복원.

사전 준비: ./data/ 에 train.jsonl, train_labels.csv 배치.

단계:
  1) train_lgbm.py            : 텍스트 모델 + 스태킹 LightGBM (CPU, 약 30분)
  2) train_strategy_c2.py ×4  : mmBERT 파인튜닝 (GPU, 시드·직렬화만 다름, 각 약 5시간)
  3) trim_vocab.py ×4         : 사용 어휘 3.8%만 유지 (예측 파리티 검증 포함)
  4) assemble_model.py        : 최종 제출물 조립 → final_submission.zip

실행:  python run_all_train.py
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_MODEL = "jhu-clsp/mmBERT-base"  # 사전학습 백본 (HuggingFace)
COMMON = ["--model", BASE_MODEL, "--epochs", "3", "--batch", "16",
          "--max-len", "512", "--lr", "3e-5", "--full"]
# 최종 모델 4개: (출력 이름, 시드, 직렬화 버전)
RUNS = [
    ("tf_s7", "7", "v3"),     # 신호토큰 직렬화
    ("tf_s77", "77", "v2"),   # ID토큰 직렬화
    ("tf_s99", "99", "v2"),   # ID토큰 직렬화 (judge)
    ("tf_s123", "123", "v3"), # 신호토큰 직렬화 (consensus)
]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


def main():
    run(["train_lgbm.py"])
    for name, seed, ser in RUNS:
        run(["train_strategy_c2.py", *COMMON, "--seed", seed, "--ser", ser,
             "--out", f"./work/{name}_full"])
        run(["trim_vocab.py", "--src", f"./work/{name}_full",
             "--dst", f"./work/{name}_trimmed", "--max-len", "512", "--ser", ser])
    run(["assemble_model.py"])
    print("\n=== 전체 파이프라인 완료: final_submission.zip ===", flush=True)


if __name__ == "__main__":
    main()
