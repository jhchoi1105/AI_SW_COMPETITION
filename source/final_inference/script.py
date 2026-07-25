# -*- coding: utf-8 -*-
"""[v3: 2-소스 앙상블] ID 강화 LGBM + mmBERT(어휘축소) — 추론.

평가 서버 사양 대응: NVIDIA T4 16GB, 3 vCPU, RAM 12GB, 추론 제한 10분, zip 1GB.

파이프라인:
  A) TF-IDF+LogReg → 확률 14
  B) LGBM ← [A확률 + 구조 49 + 표적 11 + ID 13] → 기본 예측 (전 샘플)
  C) mmBERT-base (전체 history 직렬화, max_len 512, fp16, 어휘축소+id_remap):
     온도 보정 후 LGBM과 가중 결합(0.3:0.7). 시간 마감 시 커버된 샘플만 결합.
  최종 = 가중 결합 → 클래스별 scale 보정 → argmax
  ※ xlm-r 3-소스(0.7679)와 0.0002 차이라 서버 시간 안전을 위해 2-소스 채택.

검증(세션 분리, 전체 val 튜닝): 0.7677 (held-out 추정 0.762)
"""
import csv
import json
import os
import re
import time

T_START = time.monotonic()  # 시간예산 기준점 (스크립트 시작)
# 트랜스포머 단계 마감(초). GPU면 여유롭게 끝나고, CPU 폴백 시 초과 방지용.
TRANSFORMER_DEADLINE_SEC = float(os.environ.get("TRANSFORMER_DEADLINE_SEC", 570))
TF_BATCH = 64

import joblib
import lightgbm as lgb
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
ID_RE = re.compile(r"(\d+)-step_(\d+)$")

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
ACT_IDX = {a: i for i, a in enumerate(ALL_CLASSES)}
CI_IDX = {"passed": 0, "failed": 1, "none": 2}
TIER_IDX = {"free": 0, "pro": 1, "enterprise": 2}
PREF_IDX = {"ko": 0, "en": 1, "mixed": 2}
LANG_IDX = {"py": 0, "ts": 1, "tsx": 2, "java": 3, "vue": 4, "go": 5, "rs": 6, "yaml": 7}

KEYWORDS = [
    ("kw_test", ("test", "테스트", "spec", "검증")),
    ("kw_run", ("run", "실행", "돌려", "띄워", "build", "빌드")),
    ("kw_fix", ("fix", "고쳐", "수정", "edit", "바꿔", "change", "refactor")),
    ("kw_search", ("find", "search", "찾아", "검색", "grep", "어디")),
    ("kw_read", ("open", "read", "열어", "읽어", "보여", "show", "확인")),
    ("kw_write", ("write", "create", "만들", "작성", "생성", "새 파일", "추가해")),
    ("kw_plan", ("plan", "계획", "정리해", "단계", "설계")),
    ("kw_web", ("최신", "latest", "docs", "문서 찾", "버전", "version", "릴리즈", "release")),
    ("kw_lint", ("lint", "린트", "타입", "typecheck", "type check", "포맷")),
    ("kw_dir", ("디렉터리", "디렉토리", "폴더", "directory", "구조", "목록", "list")),
]

PATH_RE = re.compile(r"[\w/\\.-]*[/\\][\w.-]+|\b[\w-]+\.(?:py|ts|tsx|js|jsx|java|go|rs|vue|css|json|yaml|yml|md|sql|toml|txt|sh|dockerfile)\b", re.I)
EXT_RE = re.compile(r"\.(?:py|ts|tsx|js|jsx|java|go|rs|vue|css|json|yaml|yml|md|sql|toml)\b", re.I)
HANGUL_RE = re.compile(r"[가-힣]")
DIR_WORDS = ("폴더", "디렉터리", "디렉토리", "구조", "목록", "folder", "directory", "structure", "tree", "layout")


# =======================
# 1. 직렬화/피처 (각 학습 코드와 완전히 동일)
# =======================

def meta_tokens(sample):
    m = sample.get("session_meta") or {}
    w = m.get("workspace") or {}
    lang_mix = w.get("language_mix") or {"none": 1}
    lang_main = max(lang_mix.items(), key=lambda x: x[1])[0]
    turn = m.get("turn_index", 0)
    open_files = w.get("open_files") or []
    return (
        f"CI_{w.get('last_ci_status', 'none')} TURN_{min(turn, 5)} "
        f"DIRTY_{w.get('git_dirty', False)} TIER_{m.get('user_tier', 'none')} "
        f"PREF_{m.get('language_pref', 'none')} "
        f"OPEN_{min(len(open_files), 3)} LANG_{lang_main}"
    )


def hist_tokens(sample):
    history = sample.get("history") or []
    acts = [h.get("name", "") for h in history if h.get("role") == "assistant_action"]
    toks = [f"PREV_{a}" for a in acts[-3:]] if acts else ["PREV_none"]
    if len(acts) >= 2:
        toks.append(f"SEQ_{acts[-2]}>{acts[-1]}")
    results = [h.get("result_summary", "") for h in history if h.get("role") == "assistant_action"]
    if results:
        toks.append("LASTRES " + str(results[-1]))
    return " ".join(toks)


def build_text(sample):
    """TF-IDF 모델 입력 (전략 A/B와 동일)."""
    prompt = sample.get("current_prompt", "")
    if not isinstance(prompt, str):
        prompt = "" if prompt is None else str(prompt)
    return prompt + " || " + hist_tokens(sample) + " " + meta_tokens(sample)


def meta_tokens_full(sample):
    m = sample.get("session_meta") or {}
    w = m.get("workspace") or {}
    lang_mix = w.get("language_mix") or {"none": 1}
    lang_main = max(lang_mix.items(), key=lambda x: x[1])[0]
    turn = m.get("turn_index", 0)
    open_files = w.get("open_files") or []
    open_str = " ".join(str(p) for p in open_files[:3])
    return (
        f"CI_{w.get('last_ci_status', 'none')} TURN_{min(turn, 5)} "
        f"DIRTY_{w.get('git_dirty', False)} TIER_{m.get('user_tier', 'none')} "
        f"PREF_{m.get('language_pref', 'none')} "
        f"OPEN_{min(len(open_files), 3)} LANG_{lang_main} FILES {open_str}"
    )


def _fmt_args(args):
    if not isinstance(args, dict) or not args:
        return ""
    vals = []
    for v in args.values():
        v = str(v)
        if len(v) > 40:
            v = v[:40]
        vals.append(v)
    return ", ".join(vals)


def build_text_full(sample):
    """트랜스포머 입력 v1 (scripts/serialize_full.py build_text_full과 동일)."""
    prompt = sample.get("current_prompt", "")
    if not isinstance(prompt, str):
        prompt = "" if prompt is None else str(prompt)
    history = sample.get("history") or []
    parts = []
    for h in reversed(history):
        if h.get("role") == "user":
            parts.append(f"U: {h.get('content', '')}")
        elif h.get("role") == "assistant_action":
            a = _fmt_args(h.get("args"))
            r = str(h.get("result_summary", ""))
            parts.append(f"A: {h.get('name', '')}({a}) -> {r}")
    hist_str = " | ".join(parts) if parts else "no_history"
    return prompt + " || " + meta_tokens_full(sample) + " || " + hist_str


def id_tokens(sample):
    """세션 일련번호 기반 토큰 (serialize_full.id_tokens와 동일)."""
    m = ID_RE.search(str(sample.get("id", "")))
    if not m:
        return "SID none"
    sess, step = int(m.group(1)), int(m.group(2))
    return (f"SID {sess} STEP {step} "
            f"M256_{sess % 256} M100_{sess % 100} M64_{sess % 64} M16_{sess % 16}")


def build_text_full_v2(sample):
    """트랜스포머 입력 v2 = ID 토큰 + v1 (serialize_full.build_text_full_v2와 동일)."""
    return id_tokens(sample) + " | " + build_text_full(sample)


def signal_tokens(sample):
    """구조 신호 전용 토큰 (serialize_full.signal_tokens와 동일)."""
    m = sample.get("session_meta") or {}
    w = m.get("workspace") or {}
    history = sample.get("history") or []
    acts = [h.get("name", "") for h in history if h.get("role") == "assistant_action"]
    results = [str(h.get("result_summary", "")) for h in history if h.get("role") == "assistant_action"]
    open_files = w.get("open_files") or []
    seq = " ".join(f"PREV_{a}" for a in acts[-3:]) if acts else "PREV_none"
    if len(acts) >= 2:
        seq += f" SEQ_{acts[-2]}>{acts[-1]}"
    last_res = results[-1].lower() if results else ""
    rflag = "RES_none"
    if last_res:
        if "fail" in last_res or "error" in last_res or "err" in last_res:
            rflag = "RES_fail"
        elif "0 match" in last_res or "no match" in last_res or "found 0" in last_res:
            rflag = "RES_zero"
        elif "pass" in last_res or last_res.startswith("ok"):
            rflag = "RES_ok"
    exts = []
    for p in open_files[:2]:
        p = str(p)
        exts.append("EXT_" + (p.rsplit(".", 1)[-1] if "." in p else "none"))
    ext_str = " ".join(exts) if exts else "EXT_none"
    return (f"CIF_{w.get('last_ci_status', 'none')} "
            f"DIRTY_{int(bool(w.get('git_dirty', False)))} "
            f"NOPEN_{min(len(open_files), 4)} {ext_str} "
            f"NACT_{min(len(acts), 6)} {rflag} {seq}")


def build_text_full_v3(sample):
    """트랜스포머 입력 v3 = ID토큰 + 구조신호 + v1 (serialize_full.build_text_full_v3와 동일)."""
    return id_tokens(sample) + " " + signal_tokens(sample) + " | " + build_text_full(sample)


SERIALIZERS = {"v1": build_text_full, "v2": build_text_full_v2, "v3": build_text_full_v3}


def build_struct(sample):
    """구조 피처 49 (strategy_b_submit/script.py와 동일)."""
    m = sample.get("session_meta") or {}
    w = m.get("workspace") or {}
    history = sample.get("history") or []
    acts = [h.get("name", "") for h in history if h.get("role") == "assistant_action"]
    results = [str(h.get("result_summary", "")) for h in history if h.get("role") == "assistant_action"]
    prompt = sample.get("current_prompt", "") or ""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    p_low = prompt.lower()

    lang_mix = w.get("language_mix") or {}
    if lang_mix:
        lang_main, lang_share = max(lang_mix.items(), key=lambda x: x[1])
    else:
        lang_main, lang_share = "none", 0.0
    open_files = w.get("open_files") or []

    last1 = ACT_IDX.get(acts[-1], 14) if len(acts) >= 1 else 14
    last2 = ACT_IDX.get(acts[-2], 14) if len(acts) >= 2 else 14
    last3 = ACT_IDX.get(acts[-3], 14) if len(acts) >= 3 else 14

    repeat = 0
    for a in reversed(acts):
        if acts and a == acts[-1]:
            repeat += 1
        else:
            break

    act_counts = [0.0] * 14
    for a in acts:
        i = ACT_IDX.get(a)
        if i is not None:
            act_counts[i] += 1.0

    last_res = results[-1].lower() if results else ""
    res_flags = [
        float("fail" in last_res),
        float("pass" in last_res or last_res.startswith("ok")),
        float("error" in last_res or "err" in last_res),
        float("0 match" in last_res or "no match" in last_res or "found 0" in last_res),
    ]
    kw_flags = [float(any(k in p_low for k in kws)) for _, kws in KEYWORDS]

    return [
        float(m.get("turn_index", 0)),
        float(m.get("elapsed_session_sec", 0)),
        float(m.get("budget_tokens_remaining", 0)),
        float(w.get("loc", 0)),
        float(bool(w.get("git_dirty", False))),
        float(len(open_files)),
        float(lang_share),
        float(CI_IDX.get(w.get("last_ci_status"), 2)),
        float(TIER_IDX.get(m.get("user_tier"), 0)),
        float(PREF_IDX.get(m.get("language_pref"), 0)),
        float(LANG_IDX.get(lang_main, 8)),
        float(last1), float(last2), float(last3),
        float(len(acts)),
        float(len(set(acts))),
        float(repeat),
        float(sum(1 for h in history if h.get("role") == "user")),
        *act_counts,
        *res_flags,
        float(len(prompt)),
        float(len(prompt.split())),
        float("?" in prompt),
        *kw_flags,
    ]


def extra_features(sample):
    """표적 피처 11 (scripts/experiment_b2.py와 동일)."""
    prompt = sample.get("current_prompt", "") or ""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    p_low = prompt.lower()
    m = sample.get("session_meta") or {}
    w = (m.get("workspace") or {})
    open_files = [str(p) for p in (w.get("open_files") or [])]
    history = sample.get("history") or []

    paths_in_prompt = PATH_RE.findall(prompt)
    hist_vals = []
    for h in history:
        if h.get("role") == "assistant_action" and isinstance(h.get("args"), dict):
            hist_vals.extend(str(v) for v in h["args"].values())
    open_basenames = [os.path.basename(p) for p in open_files]
    hangul = len(HANGUL_RE.findall(prompt))

    return [
        float(min(len(paths_in_prompt), 3)),
        float("*" in prompt),
        float(bool(EXT_RE.search(prompt))),
        float(any(b and b in prompt for b in open_basenames)),
        float(any(v and v in prompt for v in hist_vals)),
        float("`" in prompt or '"' in prompt or "'" in prompt),
        float(any(k in p_low for k in DIR_WORDS)),
        float(prompt.count("/") + prompt.count("\\")),
        float(bool(re.search(r"\bline\b|줄|:\d+", p_low))),
        float(hangul / max(len(prompt), 1)),
        float(len(history)),
    ]


def id_features(sample):
    """세션 생성 구간과 step 신호.

    sim ID에서는 세션 일련번호, au ID에서는 변형 번호가 첫 값이 된다.
    검증에서 효과가 없던 문자열 원문 대신 수치와 작은 나머지만 사용한다.
    """
    match = ID_RE.search(str(sample.get("id", "")))
    if not match:
        return [0.0] * 13
    session_no, step = (int(x) for x in match.groups())
    return [
        float(step),
        float(session_no),
        *[
            float(session_no % mod)
            for mod in (2, 3, 4, 5, 7, 10, 16, 32, 64, 100, 256)
        ],
    ]


def session_key(sample):
    return str(sample.get("id", "")).rsplit("-step_", 1)[0]


def build_lookup_overrides(samples):
    """세션 내 (user 발화 → 직후 assistant_action) 조회. 충돌 발화 제외."""
    sessions = {}
    for s in samples:
        sessions.setdefault(session_key(s), []).append(s)
    overrides = {}
    for lst in sessions.values():
        lookup, conflicts = {}, set()
        for s in lst:
            hist = s.get("history") or []
            for i, h in enumerate(hist):
                if (h.get("role") == "user" and i + 1 < len(hist)
                        and hist[i + 1].get("role") == "assistant_action"):
                    c, a = h.get("content", ""), hist[i + 1].get("name", "")
                    if not c or not a:
                        continue
                    if c in lookup and lookup[c] != a:
                        conflicts.add(c)
                    lookup[c] = a
        for s in lst:
            cp = s.get("current_prompt", "")
            if isinstance(cp, str) and cp in lookup and cp not in conflicts and lookup[cp] in ALL_CLASSES:
                overrides[s.get("id", "")] = lookup[cp]
    return overrides


# =======================
# 2. 입출력 유틸
# =======================

def load_jsonl(path):
    samples = []
    # utf-8-sig는 일반 UTF-8과 BOM 포함 UTF-8을 모두 읽는다.
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}")
    return samples


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission 컬럼이 (id, action)이 아님: {fieldnames}")
    return fieldnames, rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =======================
# 3. 트랜스포머 torch 추론 (GPU 우선, 시간예산 내 부분 처리)
# =======================

def transformer_probs_deadline(model, tokenizer, texts, device, deadline_sec,
                               max_len, remap=None):
    """짧은 샘플부터 배치 처리, 마감 시각 도달 시 중단.

    remap: 어휘 축소 모델용 [전체 id → 축소 id] 배열 (없으면 원본 id 사용).
    반환: (probs[n,14], covered[n] bool) — 처리 못한 행은 covered=False.
    """
    import torch

    enc_all = tokenizer(texts, truncation=True, max_length=max_len)
    seqs = [np.asarray(x, dtype=np.int64) for x in enc_all["input_ids"]]
    lens = [len(x) for x in seqs]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    pad_id = tokenizer.pad_token_id or 0
    probs = np.zeros((len(texts), 14), dtype=np.float64)
    covered = np.zeros(len(texts), dtype=bool)
    with torch.no_grad():
        for s in range(0, len(order), TF_BATCH):
            if time.monotonic() - T_START > deadline_sec:
                print(f" 시간 마감 도달 → 트랜스포머 중단 ({int(covered.sum())}건 처리)")
                break
            idx = order[s:s + TF_BATCH]
            # 길이를 32의 배수로 반올림 → 배치 shape 종류를 제한해 커널 재준비 오버헤드 제거
            maxlen = -(-max(lens[i] for i in idx) // 32) * 32
            ids_np = np.full((len(idx), maxlen), pad_id, dtype=np.int64)
            mask_np = np.zeros((len(idx), maxlen), dtype=np.int64)
            for r, i in enumerate(idx):
                n = lens[i]
                ids_np[r, :n] = seqs[i]
                mask_np[r, :n] = 1
            if remap is not None:
                ids_np = remap[ids_np]
            ids = torch.from_numpy(ids_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            p = torch.softmax(logits.float(), -1).cpu().numpy()
            for r, i in enumerate(idx):
                probs[i] = p[r]
                covered[i] = True
    return probs, covered


# =======================
# 4. 추론 실행
# =======================

def main():
    # 평가 서버가 script.py를 다른 working directory에서 실행할 수 있다.
    # 데이터가 script 옆에 있으면 그 위치를, 아니면 현재 cwd/data를 사용한다.
    data_candidates = [
        os.path.join(SCRIPT_DIR, "data"),
        os.path.abspath(os.path.join(os.getcwd(), "data")),
    ]
    data_dir = next(
        (
            path
            for path in data_candidates
            if os.path.isfile(os.path.join(path, "test.jsonl"))
            and os.path.isfile(os.path.join(path, "sample_submission.csv"))
        ),
        None,
    )
    if data_dir is None:
        raise FileNotFoundError(
            "data/test.jsonl 및 data/sample_submission.csv를 찾지 못했습니다. "
            f"확인한 위치: {data_candidates}"
        )

    TEST_PATH = os.path.join(data_dir, "test.jsonl")
    SAMPLE_SUB_PATH = os.path.join(data_dir, "sample_submission.csv")
    OUT_PATH = os.path.join(os.path.dirname(data_dir), "output", "submission.csv")

    print("Load models...")
    text_model = joblib.load(os.path.join(MODEL_DIR, "text_model.pkl"))
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm_final.txt"))
    with open(os.path.join(MODEL_DIR, "lgbm_classes.json"), encoding="utf-8") as f:
        lgbm_classes = json.load(f)
    with open(os.path.join(MODEL_DIR, "params.json"), encoding="utf-8") as f:
        params = json.load(f)
    lgbm_weight = float(params["lgbm_weight"])
    tf_specs = params["transformers"]  # [{dir, weight, temp, max_len}, ...] 가중치 내림차순
    blend_scale = np.array(params["scale"], dtype=np.float64)       # 최종 scale (lgbm 클래스 순서)
    lgbm_scale = np.array(params["lgbm_scale"], dtype=np.float64)   # LGBM 단독 폴백용

    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    print(f" samples={len(samples)}")

    print("LGBM branch...")
    texts_b = [build_text(s) for s in samples]
    tfidf_probs = text_model.predict_proba(texts_b)
    struct = np.array([build_struct(s) for s in samples], dtype=np.float64)
    extra = np.array([extra_features(s) for s in samples], dtype=np.float64)
    id_feats = np.array([id_features(s) for s in samples], dtype=np.float64)
    lgbm_probs = booster.predict(
        np.hstack([tfidf_probs, struct, extra, id_feats])
    )
    final_probs = lgbm_probs * lgbm_scale
    print(f" LGBM 완료 ({time.monotonic()-T_START:.0f}s 경과)")

    # ---- 트랜스포머들 (GPU 우선, 가중치 순서 처리, 실패해도 LGBM 예측 유지) ----
    tf_results = []  # (weight, probs, covered, spec)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_cache = {}  # 직렬화 버전별 캐시
        for spec in tf_specs:
            try:
                # 선택적 3-way: 앞선 두 모델이 서로 다르게 예측한 행만
                # 세 번째 모델에 보낸다. 게이트 모델이 실패하거나 시간 마감에
                # 걸리면 결합 단계에서 기존 champion 가중치로 자동 복귀한다.
                selected_idx = None
                if spec.get("gate") in ("disagreement", "disagreement_consensus"):
                    if len(tf_results) < 2:
                        raise ValueError("disagreement gate에는 앞선 모델 2개가 필요함")
                    _, p0, c0, _ = tf_results[0]
                    _, p1, c1, _ = tf_results[1]
                    gate_mask = c0 & c1 & (p0.argmax(1) != p1.argmax(1))
                    selected_idx = np.flatnonzero(gate_mask)
                    print(f" 선택 게이트: {len(selected_idx)}/{len(samples)}건 "
                          f"({len(selected_idx)/max(len(samples),1):.1%})")
                    if len(selected_idx) == 0:
                        continue

                tf_dir = os.path.join(MODEL_DIR, spec["dir"])
                with open(os.path.join(tf_dir, "classes.json"), encoding="utf-8") as f:
                    tf_classes = json.load(f)
                tokenizer = AutoTokenizer.from_pretrained(tf_dir)
                model = AutoModelForSequenceClassification.from_pretrained(
                    tf_dir, torch_dtype=torch.float16)
                model = model.to(device) if device == "cuda" else model.float()
                model.eval()
                remap_path = os.path.join(tf_dir, "id_remap.npy")
                remap = np.load(remap_path) if os.path.exists(remap_path) else None
                ser = spec.get("ser", "v1")
                if ser not in text_cache:
                    text_cache[ser] = [SERIALIZERS[ser](s) for s in samples]
                texts_c = text_cache[ser]
                texts_run = (
                    texts_c if selected_idx is None
                    else [texts_c[i] for i in selected_idx]
                )
                print(f"Transformer [{spec['dir']}] (device={device}, w={spec['weight']}, "
                      f"len={spec['max_len']}, ser={ser}, remap={'Y' if remap is not None else 'N'})")
                run_probs, run_covered = transformer_probs_deadline(
                    model, tokenizer, texts_run, device, TRANSFORMER_DEADLINE_SEC,
                    max_len=int(spec["max_len"]), remap=remap)
                if selected_idx is None:
                    probs, covered = run_probs, run_covered
                else:
                    probs = np.zeros((len(samples), 14), dtype=np.float64)
                    covered = np.zeros(len(samples), dtype=bool)
                    probs[selected_idx] = run_probs
                    covered[selected_idx] = run_covered
                reorder = [tf_classes.index(c) for c in lgbm_classes]
                probs = probs[:, reorder]
                # 온도 보정
                t = float(spec.get("temp", 1.0))
                if t != 1.0:
                    probs = probs ** t
                    probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-9, None)
                if spec.get("gate") == "disagreement_consensus":
                    if len(tf_results) < 3:
                        raise ValueError("consensus gate에는 기본 2개와 선행 judge가 필요함")
                    _, p0, c0, _ = tf_results[0]
                    _, p1, c1, _ = tf_results[1]
                    _, pj, cj, _ = tf_results[2]
                    a0, a1 = p0.argmax(1), p1.argmax(1)
                    aj, ac = pj.argmax(1), probs.argmax(1)
                    consensus = covered & c0 & c1 & cj & (ac == aj) & ((ac == a0) | (ac == a1))
                    covered &= consensus
                    print(f" 4-way 합의 게이트 통과: {int(covered.sum())}/{len(samples)}건")
                tf_results.append((float(spec["weight"]), probs, covered, spec))
                print(f" 처리 {int(covered.sum())}/{len(samples)}건 "
                      f"({time.monotonic()-T_START:.0f}s 경과)")
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f" [{spec.get('dir','?')}] 생략 (사유: {type(e).__name__}: {e})")
    except Exception as e:
        print(f" 트랜스포머 전체 생략 (사유: {type(e).__name__}: {e})")

    # ---- 가용 소스 가중 결합 (샘플별로 커버된 소스만 사용, 가중치 재정규화) ----
    if tf_results:
        n = len(samples)
        num = lgbm_weight * lgbm_probs
        den = np.full((n, 1), lgbm_weight)
        gate_covered = np.zeros(n, dtype=bool)
        for _, _, covered, spec in tf_results:
            if spec.get("gate"):
                gate_covered |= covered
        for w, probs, covered, spec in tf_results:
            cv = covered[:, None]
            sample_w = np.full(n, w, dtype=np.float64)
            if "weight_when_gate_missing" in spec:
                sample_w = np.where(
                    gate_covered,
                    w,
                    float(spec["weight_when_gate_missing"]),
                )
            sample_w = sample_w[:, None]
            num = num + np.where(cv, sample_w * probs, 0.0)
            den = den + np.where(cv, sample_w, 0.0)
        mixed = (num / den) * blend_scale
        any_tf = np.zeros(n, dtype=bool)
        for _, _, covered, _ in tf_results:
            any_tf |= covered
        final_probs = np.where(any_tf[:, None], mixed, final_probs)
        print(f" 앙상블 적용 {int(any_tf.sum())}/{n}건 ({time.monotonic()-T_START:.0f}s 경과)")

    preds = [lgbm_classes[i] for i in final_probs.argmax(axis=1)]

    # ---- 세션 내 조회 오버라이드 (train 기준 정확도 100%, 커버 없으면 무영향) ----
    overrides = build_lookup_overrides(samples)
    if overrides:
        for i, s in enumerate(samples):
            a = overrides.get(s.get("id", ""))
            if a is not None:
                preds[i] = a
    print(f"Lookup override: {len(overrides)}건")

    bad = {p for p in preds if p not in ALL_CLASSES}
    if bad:
        raise ValueError(f"허용되지 않는 클래스 예측: {bad}")

    print("Build submission...")
    ids = [s.get("id", "") for s in samples]
    fieldnames, sub_rows = load_sample_submission(SAMPLE_SUB_PATH)
    pred_map = dict(zip(ids, preds))
    n_missing = 0
    for row in sub_rows:
        p = pred_map.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 id {n_missing}건")
    save_submission(OUT_PATH, fieldnames, sub_rows)
    print(f"Saved: {OUT_PATH} (rows={len(sub_rows)})")


if __name__ == "__main__":
    main()
