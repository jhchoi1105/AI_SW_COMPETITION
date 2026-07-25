# 학습 코드 제출물 — 코딩 에이전트 다음 행동 예측 (Team MOOD)

Private Score **0.7967670316** 을 기록한 최종 제출물(`g4_c15`)의 **전체 학습 파이프라인**입니다.
추론 코드는 Private Score를 기록한 제출물의 코드로 대체되며, 동일한 사본을 `final_inference/`에 포함했습니다.

---

## 1. 개발 환경

| 항목 | 내용 |
|---|---|
| OS | Windows 11 Pro (빌드 10.0.26200) |
| Python | 3.12.10 |
| GPU | NVIDIA GeForce RTX 4060 8GB (학습) / 평가 서버 T4 (추론) |
| CUDA | 12.1 (torch 배포판 기준) |

### 라이브러리 버전 (학습 환경)

```
torch==2.5.1+cu121
transformers==5.9.0
lightgbm==4.6.0
scikit-learn==1.8.0
numpy==2.4.3
joblib==1.5.3
```

- 추론(평가 서버) 의존성은 `final_inference/requirements.txt` 참조 (Private 제출물과 동일).

## 2. 학습·개발 활용 자원 출처

| 자원 | 출처 | 용도 |
|---|---|---|
| **jhu-clsp/mmBERT-base** | HuggingFace Hub (https://huggingface.co/jhu-clsp/mmBERT-base) | 트랜스포머 4개의 사전학습 백본 (파인튜닝) |
| 대회 제공 데이터 (train.jsonl, train_labels.csv) | 대회 페이지 | 유일한 학습 데이터 |

- **외부 데이터는 일절 사용하지 않았습니다** (학습은 제공된 train 70,000건만 사용).
- 사전학습 백본 가중치는 규정(오프라인 평가)에 따라 어휘 축소·FP16 변환 후 제출 zip의 `model/`에 포함했습니다.

## 3. 실행 방법

```
code_submission/
├── data/                  ← 여기에 대회 제공 train.jsonl, train_labels.csv 배치
├── run_all_train.py       ← 전체 파이프라인 실행 (아래 1)~4) 순차 수행)
├── train_lgbm.py          ← 1) 텍스트 모델 + 스태킹 LightGBM
├── train_strategy_c2.py   ← 2) mmBERT 파인튜닝 (시드·직렬화 인자만 다르게 4회)
├── serialize_full.py      ←    입력 직렬화 (v2 ID토큰 / v3 신호토큰)
├── trim_vocab.py          ← 3) 사용 어휘 3.8%만 유지 (예측 파리티 검증 포함)
├── assemble_model.py      ← 4) 최종 제출물 조립 → final_submission.zip
├── final_params.json      ←    앙상블 하이퍼파라미터 (Private 기록본과 동일)
└── final_inference/       ←    Private 제출물의 추론 코드 사본 (script.py, requirements.txt)
```

```bash
cd code_submission
python run_all_train.py     # 전체 실행 (RTX 4060 기준 총 약 21시간)
```

개별 실행 시 순서 (run_all_train.py가 수행하는 명령과 동일):

```bash
# 1) LightGBM 파이프라인 (CPU, 약 30분)
python train_lgbm.py

# 2)+3) 트랜스포머 4개: 파인튜닝(각 약 5시간) 후 어휘 축소
python train_strategy_c2.py --model jhu-clsp/mmBERT-base --epochs 3 --batch 16 --max-len 512 --lr 3e-5 --full --seed 7   --ser v3 --out ./work/tf_s7_full
python trim_vocab.py --src ./work/tf_s7_full --dst ./work/tf_s7_trimmed --max-len 512 --ser v3
#   (동일 명령을 --seed 77/--ser v2, --seed 99/--ser v2, --seed 123/--ser v3 으로 반복,
#    출력 폴더는 tf_s77 / tf_s99 / tf_s123)

# 4) 최종 조립 → ./final_submission.zip (Private 제출물과 동일 구조)
python assemble_model.py
```

## 4. 최종 모델 구성 (조립 결과)

| 구성 요소 | 학습 스크립트 | 설정 |
|---|---|---|
| LightGBM | train_lgbm.py | 텍스트 OOF 14 + 구조 49 + 표적 11 + ID 13 피처, n_estimators=500, seed 42 |
| mmBERT tf_s77 | train_strategy_c2.py | seed 77, v2(ID토큰) 직렬화, 3ep·len512·bs16·lr3e-5 |
| mmBERT tf_s7 | train_strategy_c2.py | seed 7, v3(신호토큰) 직렬화, 동일 |
| mmBERT tf_s99 | train_strategy_c2.py | seed 99, v2 직렬화, 동일 — 이견(judge) 게이트 |
| mmBERT tf_s123 | train_strategy_c2.py | seed 123, v3 직렬화, 동일 — 합의(consensus) 게이트 |
| 앙상블 결합 | final_params.json | LGBM 0.10 / s77 0.45 / s7 0.25 / s99 0.20(gate) / s123 0.15(consensus), temp 0.28 |

## 5. 재현성 관련 노트

- 모든 난수 시드 고정: LightGBM `random_state=42`, 트랜스포머 시드 7/77/99/123 (torch·numpy·python random 모두 스크립트 내 고정).
- 학습·추론의 피처 정의 일치: `train_lgbm.py`는 피처 함수를 **최종 추론 코드(final_inference/script.py)에서 직접 import** 하므로 학습·추론 간 피처 불일치가 원천적으로 불가능합니다. 직렬화(serialize_full.py)도 추론 코드와 동일 정의이며, `trim_vocab.py`가 어휘 축소 전후 **예측 완전 일치(파리티)** 를 검증합니다.
- GPU 부동소수점 연산의 비결정성으로 트랜스포머 재학습 시 소수점 미세 차이가 발생할 수 있습니다 (대회 규정의 허용 범위 내).
- 학습 코드·주석 인코딩: UTF-8. 모든 경로는 상대 경로.
