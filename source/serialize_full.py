# -*- coding: utf-8 -*-
"""전체 history 직렬화 — 전략 C 강화판 입력.

구성: current_prompt || 메타 토큰 || 최근순 history (사용자 발화 + 행동(인자)->결과)
- 최근 것이 앞에 오도록 역순 배치 → max_len 초과 시 오래된 턴부터 잘림
- 행동 인자는 값만 뽑아 40자 제한 (경로/패턴이 탐색 계열 구분의 단서)
"""


def meta_tokens(sample):
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
    prompt = sample.get("current_prompt", "")
    if not isinstance(prompt, str):
        prompt = "" if prompt is None else str(prompt)

    history = sample.get("history") or []
    parts = []
    for h in reversed(history):  # 최근 턴이 앞으로
        if h.get("role") == "user":
            parts.append(f"U: {h.get('content', '')}")
        elif h.get("role") == "assistant_action":
            a = _fmt_args(h.get("args"))
            r = str(h.get("result_summary", ""))
            parts.append(f"A: {h.get('name', '')}({a}) -> {r}")
    hist_str = " | ".join(parts) if parts else "no_history"

    return prompt + " || " + meta_tokens(sample) + " || " + hist_str


import re as _re
_ID_RE = _re.compile(r"(\d+)-step_(\d+)$")


def id_tokens(sample):
    """세션 일련번호 기반 토큰 — LGBM ID 피처(중요도 최상위)를 트랜스포머에도 노출."""
    m = _ID_RE.search(str(sample.get("id", "")))
    if not m:
        return "SID none"
    sess, step = int(m.group(1)), int(m.group(2))
    return (f"SID {sess} STEP {step} "
            f"M256_{sess % 256} M100_{sess % 100} M64_{sess % 64} M16_{sess % 16}")


def build_text_full_v2(sample):
    """v1 + ID 토큰 (직렬화 맨 앞 메타에 포함)."""
    return id_tokens(sample) + " | " + build_text_full(sample)


def signal_tokens(sample):
    """LGBM이 강하게 쓰는 구조 신호를 트랜스포머용 전용 토큰으로 (v3)."""
    m = sample.get("session_meta") or {}
    w = m.get("workspace") or {}
    history = sample.get("history") or []
    acts = [h.get("name", "") for h in history if h.get("role") == "assistant_action"]
    results = [str(h.get("result_summary", "")) for h in history if h.get("role") == "assistant_action"]
    open_files = w.get("open_files") or []

    # 직전 행동 시퀀스 (최근 3개)
    seq = " ".join(f"PREV_{a}" for a in acts[-3:]) if acts else "PREV_none"
    if len(acts) >= 2:
        seq += f" SEQ_{acts[-2]}>{acts[-1]}"

    # 마지막 result의 결과 플래그
    last_res = results[-1].lower() if results else ""
    rflag = "RES_none"
    if last_res:
        if "fail" in last_res or "error" in last_res or "err" in last_res:
            rflag = "RES_fail"
        elif "0 match" in last_res or "no match" in last_res or "found 0" in last_res:
            rflag = "RES_zero"
        elif "pass" in last_res or last_res.startswith("ok"):
            rflag = "RES_ok"

    # 열린 파일 확장자 (최대 2개)
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
    """v2(ID토큰) + 구조 신호 토큰."""
    return id_tokens(sample) + " " + signal_tokens(sample) + " | " + build_text_full(sample)
