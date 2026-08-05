# AI·SW 중심대학 디지털 경진대회

코딩 에이전트 세션에서 다음 행동을 14개 클래스 중 하나로 예측한 대회의 최종 자료입니다.

## 결과

| 항목 | 내용 |
|---|---|
| Private Macro-F1 | **0.7967670316** |
| 최고점 제출물 | `g4_c15.zip` |
| 모델 | LightGBM + mmBERT 4모델 게이트 앙상블 |
| 평가 환경 | T4 16GB, 3 vCPU, 10분, ZIP 1GB 이하 |

GitHub 저장소에는 소스와 문서를, [Releases](../../releases)에는 893.6MiB 최종 모델
`g4_c15.zip`을 배포합니다. 로컬 `GitHub_최종본`에는 Release 자산도 함께 보존합니다.

## 구성

```text
GitHub_최종본/
├── README.md
├── MODEL_CARD.md
├── SHA256SUMS.txt
├── source/                 # 최종 학습·추론·조립 코드
├── docs/
│   ├── strategy/           # 대회 전략·일정·정리
│   ├── presentation/       # 발표 자료와 대본
│   └── experiments/        # 규정과 추가 백본 실험
├── artifacts/
│   └── code_submission.zip
└── release_assets/
    └── g4_c15.zip
```

## 최종 앙상블

- LightGBM `0.10`
- mmBERT seed77/v2 `0.45`
- mmBERT seed7/v3 `0.25` (`gate` 미적용 시 `0.45`)
- mmBERT seed99/v2 disagreement judge `0.20`
- mmBERT seed123/v3 consensus `0.15`
- Transformer temperature `0.28`

세부 파라미터는 [`source/final_params.json`](source/final_params.json), 재현 방법은
[`source/README.md`](source/README.md)를 참고하세요.

## 재현

대회 제공 `train.jsonl`, `train_labels.csv`를 `source/data/`에 배치한 후 실행합니다.

```bash
cd source
python run_all_train.py
```

RTX 4060 기준 약 21시간이 필요합니다. 대회 데이터는 재배포하지 않습니다.

## 주요 문서

- [대회 전략 전체 기록](docs/strategy/대회_전략.md)
- [대회 정리](docs/strategy/대회_정리.md)
- [발표 자료](docs/presentation/발표_정리.md)
- [발표 대본](docs/presentation/발표_대본.md)
- [발표 공부 노트](docs/presentation/발표_공부노트.md) — 용어 사전부터 Q&A까지 발표자 학습용
  ([인쇄용 PDF](docs/presentation/발표_공부노트.pdf), A4 28쪽)
- [포스터 세션 초안](docs/presentation/포스터_초안.md) — 포스터 1장 문안과 배치

별도 라이선스는 지정하지 않았으며, 대회 데이터와 사전학습 모델은 원래 이용 조건을 따릅니다.
