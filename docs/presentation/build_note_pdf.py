# -*- coding: utf-8 -*-
"""발표_공부노트.md → 인쇄용 HTML (이후 브라우저로 PDF 인쇄).

인쇄를 위한 조정:
  - <details>를 전부 펼침 (접힌 답은 종이에서 볼 수 없음)
  - "- [ ]" 체크박스를 ☐ 로 치환하고 불릿 제거
  - 퀴즈 답안은 문제와 다른 페이지로 분리 (가리고 풀 수 있게)
  - Part 경계마다 페이지 나눔, 표·코드블록 내부 분할 금지
  - 표지와 목차 자동 생성

사용법:
    pip install markdown
    python build_note_pdf.py 발표_공부노트.md note.html
    msedge --headless --no-pdf-header-footer \
           --print-to-pdf=발표_공부노트.pdf "file:///<절대경로>/note.html"

미리보기: note.html?only=N  → N번째 섹션만 표시 (0=Part 0 앞, 1=Part 0, 2=Part 1 …)
"""
import io
import re
import sys
from pathlib import Path

import markdown

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

text = SRC.read_text(encoding="utf-8")

# ── 인쇄용 전처리 ──────────────────────────────────────────────
# 1) 문서 상단의 "이 문서의 목적" 안내는 인쇄본 표지로 따로 뽑으므로 본문에서 제거
text = re.sub(r"\A# 발표 준비 완전 학습 노트\n\n(> .*\n)+\n", "", text)

# 2) details/summary → 항상 보이는 블록으로 변환
text = text.replace("<details open>", '<div class="disc" markdown="1">').replace(
    "<details>", '<div class="disc" markdown="1">'
)
text = text.replace("</details>", "</div>")
text = re.sub(
    r"<summary>(.*?)</summary>", r'<p class="disc-title">\1</p>', text, flags=re.S
)

# 3) 체크박스
text = re.sub(r"^(\s*)- \[ \] ", r"\1- ☐ ", text, flags=re.M)

# 3-1) 문단 바로 다음 줄에서 시작하는 목록은 Python-Markdown이 문단에 이어붙인다.
#      코드펜스 밖에서, 들여쓰기 없는 문단 뒤에 오는 최상위 목록 앞에 빈 줄을 넣는다.
TOP_ITEM = re.compile(r"^(?:[-*+]|\d+\.)\s")
ANY_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
fixed, fence = [], False
for ln in text.split("\n"):
    if ln.lstrip().startswith("```"):
        fence = not fence
    elif not fence and TOP_ITEM.match(ln) and fixed:
        prev = fixed[-1]
        if prev.strip() and not prev[:1].isspace() and not ANY_ITEM.match(prev):
            fixed.append("")
    fixed.append(ln)
text = "\n".join(fixed)

# 4) 로컬 문서 링크는 종이에서 눌리지 않으므로 텍스트만 남김
text = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"\1", text)

html_body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "md_in_html", "sane_lists"],
)

# ── 목차 생성 + Part 페이지 나눔 ────────────────────────────────
parts = re.findall(r"<h1>(Part \d+\..*?)</h1>", html_body)
toc = "\n".join(
    f'<li><span class="tnum">{p.split(".")[0]}</span>'
    f'<span class="ttl">{p.split(".", 1)[1].strip()}</span></li>'
    for p in parts
)

# 체크박스 항목은 불릿 없이 ☐만 보이게
html_body = html_body.replace("<li>☐ ", '<li class="cb">☐ ')

# 퀴즈 답안은 문제와 다른 페이지에 오도록 분리 (가리고 풀 수 있게)
html_body = re.sub(
    r'<div class="disc">(\s*<p class="disc-title"><b>답 확인)',
    r'<div class="disc answers">\1',
    html_body,
)

# h1 단위로 <section>을 감싸 페이지 나눔과 미리보기를 함께 제어
chunks = re.split(r"(?=<h1>)", html_body)
sections = []
for i, ch in enumerate(chunks):
    if not ch.strip():
        continue
    cls = "part" + ("" if not sections else " pb")
    sections.append(f'<section class="{cls}" id="s{i}">{ch}</section>')
html_body = "\n".join(sections)

# 미리보기 모드: ?only=N 이면 해당 섹션만 표시 (인쇄 결과에는 영향 없음)
PREVIEW_JS = """
<script>
const only = new URLSearchParams(location.search).get('only');
if (only !== null) {
  document.querySelectorAll('.cover,.toc').forEach(e => e.remove());
  document.querySelectorAll('section.part').forEach((s, i) => {
    if (String(i) !== only) s.remove(); else s.classList.remove('pb');
  });
}
</script>"""

CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }

* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: 'Malgun Gothic', 'Pretendard', 'Noto Sans KR', sans-serif;
  font-size: 10.2pt; line-height: 1.62; color: #14181F;
  margin: 0; word-break: keep-all; overflow-wrap: anywhere;
}

/* ── 표지 ───────────────────────────────── */
.cover { height: 258mm; display: flex; flex-direction: column; justify-content: center;
         page-break-after: always; }
.cover .eyebrow { font-size: 11pt; letter-spacing: .18em; color: #7A8494; margin-bottom: 10mm; }
.cover h1 { font-size: 30pt; line-height: 1.28; margin: 0 0 7mm; border: 0; padding: 0; }
.cover .sub { font-size: 12pt; color: #46505F; line-height: 1.75; margin-bottom: 14mm; }
.cover .meta { border-top: 2px solid #14181F; padding-top: 5mm; font-size: 10pt; color: #46505F; }
.cover .meta b { color: #14181F; }
.cover .how { margin-top: 12mm; background: #F4F6F9; border-left: 4px solid #1D4ED8;
              padding: 5mm 6mm; font-size: 9.6pt; line-height: 1.75; color: #14181F; }
.cover .how b { color: #1D4ED8; }
.cover .meta, .cover .sub { color: #46505F; }

/* ── 목차 ───────────────────────────────── */
.toc { page-break-after: always; }
.toc h2 { font-size: 17pt; border: 0; margin: 0 0 8mm; }
.toc ol { list-style: none; padding: 0; margin: 0; }
.toc li { display: flex; gap: 6mm; padding: 3.6mm 0; border-bottom: 1px solid #E3E8EF;
          font-size: 11.5pt; }
.toc .tnum { flex: 0 0 20mm; font-weight: 700; color: #1D4ED8; }
.toc .ttl { flex: 1; }

/* ── 제목 ───────────────────────────────── */
h1 { font-size: 19pt; margin: 0 0 6mm; padding-bottom: 3mm;
     border-bottom: 2.5px solid #14181F; line-height: 1.35; }
section.pb { page-break-before: always; }
h2 { font-size: 14pt; margin: 9mm 0 3.5mm; padding-left: 3mm;
     border-left: 4px solid #1D4ED8; line-height: 1.4; page-break-after: avoid; }
h3 { font-size: 11.6pt; margin: 6.5mm 0 2.5mm; color: #1F2937; page-break-after: avoid; }
h1 + h2, h1 + p + h2 { margin-top: 5mm; }

p { margin: 0 0 3mm; }
strong { font-weight: 700; }
a { color: #14181F; text-decoration: none; }

/* ── 목록 ───────────────────────────────── */
ul, ol { margin: 0 0 3.5mm; padding-left: 6.5mm; }
li { margin-bottom: 1.4mm; }
li > ul, li > ol { margin-top: 1.4mm; margin-bottom: 0; }

/* ── 표 ─────────────────────────────────── */
table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm;
        font-size: 9.3pt; page-break-inside: avoid; }
th { background: #EEF2F7; text-align: left; font-weight: 700;
     border-bottom: 1.5px solid #94A3B8; padding: 2.2mm 2.6mm; }
td { border-bottom: 1px solid #E3E8EF; padding: 2.2mm 2.6mm; vertical-align: top; }
tr:nth-child(even) td { background: #FAFBFD; }

/* ── 코드 ───────────────────────────────── */
code { font-family: 'D2Coding', Consolas, monospace; font-size: 8.9pt;
       background: #F1F5F9; padding: 0.4mm 1.2mm; border-radius: 2px; }
pre { background: #F8FAFC; border: 1px solid #DDE3EC; border-left: 3px solid #94A3B8;
      border-radius: 4px; padding: 3mm 4mm; margin: 3mm 0 5mm;
      page-break-inside: avoid; overflow: hidden; }
pre code { background: none; padding: 0; font-size: 8.5pt; line-height: 1.5;
           white-space: pre-wrap; }

/* ── 인용 ───────────────────────────────── */
blockquote { margin: 3mm 0 4mm; padding: 2.6mm 4mm; background: #F4F6F9;
             border-left: 4px solid #94A3B8; font-size: 9.6pt;
             page-break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }

/* ── 용어사전 / 퀴즈 답 블록 ─────────────── */
.disc { border: 1px solid #DDE3EC; border-radius: 5px; padding: 3.5mm 4.5mm;
        margin: 3mm 0 5mm; background: #FCFDFE; page-break-inside: avoid; }
.disc-title { font-weight: 700; font-size: 11pt; margin: 0 0 2.5mm;
              padding-bottom: 1.8mm; border-bottom: 1px solid #E3E8EF; }
.disc p:last-child, .disc ol:last-child, .disc ul:last-child { margin-bottom: 0; }
.disc.answers { page-break-before: always; page-break-inside: auto; }
li.cb { list-style: none; margin-left: -4mm; }

hr { border: 0; border-top: 1px solid #E3E8EF; margin: 6mm 0; }

/* 제목 뒤에서 페이지가 끊기지 않도록 */
h1, h2, h3 { break-after: avoid-page; }
"""

COVER = """
<section class="cover">
  <div class="eyebrow">2026 AI·SW중심대학 디지털 경진대회 · AI부문</div>
  <h1>발표 준비<br>완전 학습 노트</h1>
  <div class="sub">
    코딩 에이전트 다음 행동 예측 · Team MOOD<br>
    용어 사전부터 예상 질문 대응까지, 발표자가 혼자 공부할 수 있게 정리한 자료
  </div>
  <div class="meta">
    최종 성적 <b>Private Macro-F1 0.7967670316</b> &nbsp;·&nbsp;
    제출물 <b>g4_c15.zip</b> &nbsp;·&nbsp;
    LightGBM 1 + mmBERT 4 게이트 앙상블
  </div>
  <div class="how">
    <b>공부 순서</b> — Part 0 → Part 1 → <b>Part 2(용어 사전, 가장 중요)</b> → Part 3 →
    Part 4 → Part 5(슬라이드별) → Part 6(Q&amp;A) → Part 7(암기 카드) → Part 8(퀴즈)<br><br>
    Part 2를 이해하지 못한 채 Part 5로 건너뛰면 대본을 외우게 될 뿐입니다.
    <b>용어 사전에 30분을 쓰는 게 대본을 세 번 읽는 것보다 낫습니다.</b><br><br>
    Part 8의 퀴즈는 답을 가리고 <b>소리 내어</b> 답해 보세요. 막히는 항목이 그날 공부할 부분입니다.
  </div>
</section>
"""

TOC = f"""
<section class="toc">
  <h2>목차</h2>
  <ol>{toc}</ol>
</section>
"""

out = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>발표 준비 완전 학습 노트</title>
<style>{CSS}</style></head>
<body>{COVER}{TOC}{html_body}{PREVIEW_JS}</body></html>"""

DST.write_text(out, encoding="utf-8")
print(f"HTML 생성 완료: {DST}  ({len(out):,} bytes, Part {len(parts)}개)")
