"""시편 배경 제거기 · Streamlit 진입점 (UI만 담당).

실행:  python -m streamlit run app.py
처리 로직은 core/ 에 있고, 여기서는 캐시와 화면 구성만 담당한다.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from core import config
from core.io_utils import (build_zip, image_hash, load_image, make_thumbnail, output_name,
                           save_to_folder, unique_stems)
from core.layout import SIZE_MODES, even_size, fit_scale, optimal_size_from_dims
from core.render import crop_dims, render_full, render_preview, scaled_mask
from core.segment import (METHOD_AI, SCOPE_WITH_TAPE, SCOPES, ai_detect_outline, outline_overlay,
                          segment_work)

st.set_page_config(page_title="시편 배경 제거기", page_icon="🧪", layout="wide")

st.html("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  h1 { font-size: 1.9rem !important; letter-spacing: -0.02em; }
  [data-testid="stFileUploaderDropzone"] { min-height: 150px; border: 2px dashed #B8C6D8;
      background: #F7F9FC; align-items: center; }
  [data-testid="stFileUploaderDropzone"]:hover { border-color: #2563A8; background: #EEF4FB; }
  [data-testid="stMetricValue"] { font-size: 1.35rem; }
  [data-testid="stImageCaption"] { font-size: 0.8rem; }
  .step-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 4px; }
  .step { flex: 1 1 200px; background: #F3F6FA; border: 1px solid #DCE3EC; border-radius: 10px;
          padding: 14px 16px; font-size: 0.92rem; color: #374151; }
  .step b { display: block; color: #2563A8; margin-bottom: 4px; }
</style>
""")


# ── 캐시 (PRD N-02) ─────────────────────────────────────────────
# 인자 이름 앞의 '_'는 Streamlit이 해시하지 않는 인자. 캐시 키는 이미지 해시 + 모델 + 처리 옵션.
# 후처리 슬라이더 값은 여기에 넣지 않는다 → 슬라이더를 움직여도 API를 다시 부르지 않음.
THUMB_SIDE = 900   # 미리보기용 축소본 최대 변 (실시간 처리는 이 해상도에서만 한다, B-05)


@st.cache_data(show_spinner=False, max_entries=200)
def cached_outline(img_hash: str, _data: bytes, model: str, scope: str) -> dict:
    return ai_detect_outline(_data, model, scope=scope)


@st.cache_data(show_spinner=False, max_entries=100)
def cached_segment(img_hash: str, _data: bytes, use_ai: bool, model: str, iters: int, scope: str):
    """(작업 해상도 마스크, 원본 크기, 처리 방식, 설명). 마스크는 긴 변 1600px 이하 (B-06).

    scope(시편 범위)는 AI에게 요청하는 내용 자체가 달라지므로 캐시 키에 들어간다.
    """
    return segment_work(_data, use_ai, model, iters,
                        outline_fn=lambda d, m: cached_outline(img_hash, d, m, scope))


@st.cache_data(show_spinner=False, max_entries=100)
def cached_thumb(img_hash: str, _data: bytes):
    """미리보기용 축소본. 원본 디코딩을 슬라이더 조작마다 반복하지 않는다 (B-05)."""
    return make_thumbnail(load_image(_data), THUMB_SIDE)


# ── 설정 기본값 / 되돌리기 ────────────────────────────────────────
DEFAULTS = {
    "use_ai": config.has_api_key(), "scope": SCOPE_WITH_TAPE, "iters": 5,
    "shrink": 1, "feather": 1,
    "size_mode": "중앙값", "margin": 5, "custom": False, "cw": 1200, "ch": 800,
    "bg_mode": "투명", "bg_color": "#FFFFFF", "fmt": "PNG", "density": "보통",
}
SEGMENTED = ("scope", "size_mode", "bg_mode", "fmt", "density")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
for k in SEGMENTED:                      # 위젯을 만들기 전이므로 여기서는 값을 바꿔도 된다
    if st.session_state.get(k) is None:
        st.session_state[k] = st.session_state.get(f"_prev_{k}", DEFAULTS[k])


def keep_selection(k: str):
    """segmented_control은 선택된 버튼을 다시 누르면 None이 된다.
    on_change 콜백은 위젯 생성 전에 실행되므로 여기서 직전 값으로 되돌린다 (B-01)."""
    if st.session_state.get(k) is None:
        st.session_state[k] = st.session_state.get(f"_prev_{k}", DEFAULTS[k])


def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def clear_cache():
    cached_segment.clear()
    cached_outline.clear()
    st.toast("캐시를 지웠습니다. 사진을 다시 처리합니다.", icon="🔄")


def ss_use_ai() -> bool:
    return bool(st.session_state.get("use_ai")) and config.has_api_key()


# ── 사이드바 ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    if config.has_api_key():
        st.badge(f"AI 사용 가능 · {config.MODEL}", icon=":material/check_circle:", color="green")
    else:
        st.badge("API 키 없음 · 로컬 방식만 사용", icon=":material/warning:", color="orange")
        st.caption(".env 파일에 `OPENAI_API_KEY=...`를 넣으면 AI로 테두리를 추출합니다.")

    with st.expander("🤖 AI 설정", expanded=True):
        st.toggle("AI로 테두리 추출", key="use_ai", disabled=not config.has_api_key(),
                  help="끄면 OpenCV GrabCut만으로 처리합니다. API 비용이 들지 않지만 배경이 복잡하면 정확도가 떨어집니다.")
        st.segmented_control("시편 범위", SCOPES, key="scope", width="stretch",
                             on_change=keep_selection, args=("scope",), disabled=not ss_use_ai(),
                             help="테이프 포함: 마스킹 테이프·프레임과 그 위의 라벨까지 시편으로 잘라냅니다. "
                                  "금속 노출부만: 테이프 안쪽의 맨 표면만 잘라냅니다. "
                                  "바꾸면 AI를 다시 호출합니다(사진당 1회, 이후 캐시). AI를 끄면 적용되지 않습니다.")
        st.slider("GrabCut 반복 횟수", 1, 10, key="iters",
                  help="경계를 다듬는 반복 횟수입니다. 높을수록 정밀하지만 처리 시간이 늘어납니다. 바꾸면 사진을 다시 처리합니다.")
        st.button("캐시 지우고 다시 처리", icon=":material/refresh:", on_click=clear_cache,
                  width="stretch", help="AI 오류가 났던 사진을 다시 시도할 때 사용합니다. API를 다시 호출합니다.")

    with st.expander("✂️ 경계 미세조정", expanded=True):
        st.slider("경계 축소(+) / 확장(−)", -15, 15, key="shrink", format="%d px",
                  help="+는 테두리를 안쪽으로 줄여 배경이 묻어나는 것을 없애고, −는 바깥쪽으로 넓혀 잘린 부분을 되살립니다.")
        st.slider("가장자리 부드럽게", 0, 10, key="feather", format="%d px",
                  help="테두리를 살짝 흐리게 해서 계단 현상을 줄입니다. 0이면 딱 떨어지는 경계입니다.")

    with st.expander("📐 출력 크기", expanded=True):
        st.segmented_control("기준 크기", SIZE_MODES, key="size_mode", width="stretch",
                             on_change=keep_selection, args=("size_mode",),
                             help="최소: 어떤 사진도 확대하지 않음 · 중앙값: 화질과 일관성의 균형 · 최대: 가장 큰 사진 기준 (작은 사진은 확대됨)")
        st.slider("여백", 0, 30, key="margin", format="%d%%",
                  help="시편과 이미지 가장자리 사이의 여백 비율입니다.")
        st.checkbox("출력 크기 직접 지정", key="custom")
        if st.session_state.custom:
            c1, c2 = st.columns(2)
            c1.number_input("너비 (px)", 64, 8000, step=2, key="cw", help="홀수를 입력하면 짝수로 맞춥니다.")
            c2.number_input("높이 (px)", 64, 8000, step=2, key="ch", help="홀수를 입력하면 짝수로 맞춥니다.")

    with st.expander("🎨 배경 · 형식", expanded=True):
        st.segmented_control("배경", ["투명", "흰색", "검정", "직접 선택"], key="bg_mode", width="stretch",
                             on_change=keep_selection, args=("bg_mode",))
        if st.session_state.bg_mode == "직접 선택":
            st.color_picker("배경색", key="bg_color")
        st.segmented_control("저장 형식", ["PNG", "JPG"], key="fmt", width="stretch",
                             on_change=keep_selection, args=("fmt",),
                             help="투명 배경은 PNG에서만 유지됩니다. JPG는 흰색 배경 위에 합성됩니다.")

    st.button("기본값으로 되돌리기", icon=":material/restart_alt:", on_click=reset_defaults,
              width="stretch", type="tertiary")

ss = st.session_state
use_ai, iters, scope = ss_use_ai(), ss.iters, ss.scope
shrink, feather, size_mode, margin = ss.shrink, ss.feather, ss.size_mode, ss.margin / 100
fmt, bg_mode = ss.fmt, ss.bg_mode
if bg_mode == "직접 선택":
    hexc = ss.bg_color.lstrip("#")
    bg = tuple(int(hexc[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
else:
    bg = {"투명": (0, 0, 0, 0), "흰색": (255, 255, 255, 255), "검정": (0, 0, 0, 255)}[bg_mode]
for k in SEGMENTED:                      # 선택 해제 시 되돌릴 직전 값 (위젯 key가 아니라서 언제든 쓸 수 있음)
    ss[f"_prev_{k}"] = ss[k]

# ── 헤더 & 업로드 (PRD F-01~F-04) ─────────────────────────────
st.title("🧪 시편 배경 제거기")
st.caption("사진 속 시편의 테두리를 찾아 배경을 지우고, 모든 결과를 같은 크기로 맞춰 저장합니다.")

files = st.file_uploader(
    "시편 사진 업로드",
    type=config.SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="여러 장을 한 번에 올리면 모두 같은 크기로 맞춰집니다.",
    label_visibility="collapsed",
)

if not files:
    st.html(f"""
    <div class="step-row">
      <div class="step"><b>① 사진 올리기</b>위 영역에 사진을 끌어다 놓거나 클릭해서 선택하세요.
        여러 장도 한 번에 올릴 수 있습니다.<br><small>지원 형식: {', '.join(t.upper() for t in config.SUPPORTED_TYPES)}</small></div>
      <div class="step"><b>② 결과 확인·조정</b>시편 테두리가 자동으로 추출됩니다.
        왼쪽 설정을 바꾸면 미리보기가 바로 바뀝니다.</div>
      <div class="step"><b>③ 저장</b>모든 사진이 같은 크기로 맞춰집니다.
        ZIP으로 받거나 output 폴더에 저장하세요.</div>
    </div>""")
    st.stop()

total_mb = sum(f.size for f in files) / 1024 / 1024
st.caption(f"📁 {len(files)}장 · {total_mb:.1f} MB")

# ── 분할 (캐시됨) ──────────────────────────────────────────────
items, failed = [], []
stems = unique_stems([f.name for f in files])
progress = st.progress(0.0, text="시편 테두리 추출 준비 중…")
for i, (f, stem) in enumerate(zip(files, stems)):
    progress.progress(i / len(files), text=f"테두리 추출 중 ({i + 1}/{len(files)}) · {f.name}")
    data = f.getvalue()
    try:
        h = image_hash(data)
        thumb = cached_thumb(h, data)
        work_mask, orig_size, method, detail = cached_segment(h, data, use_ai, config.MODEL, iters, scope)
        items.append({"idx": i, "stem": stem, "name": f.name, "data": data, "thumb": thumb,
                      "mask": work_mask, "orig": orig_size, "method": method, "detail": detail})
    except Exception as e:
        failed.append((f.name, type(e).__name__))
progress.empty()

if failed:
    st.error("다음 파일은 이미지로 열 수 없어 건너뛰었습니다. 파일이 손상되지 않았는지 확인해 주세요.\n\n"
             + "\n".join(f"- {escape(n)} ({err})" for n, err in failed))
if not items:
    st.stop()

ai_errors = [it for it in items if "오류" in it["detail"]]
not_found = [it for it in items if "찾지 못해" in it["detail"]]
if ai_errors:
    st.warning(f"AI 호출에 실패한 사진 {len(ai_errors)}장은 로컬 방식으로 처리했습니다. "
               "네트워크나 API 키를 확인한 뒤 사이드바의 '캐시 지우고 다시 처리'를 눌러 보세요.",
               icon=":material/cloud_off:")
if not_found:
    st.info(f"AI가 시편을 찾지 못한 사진 {len(not_found)}장은 로컬 방식으로 처리했습니다. "
            "비교 보기에서 테두리를 확인해 주세요.", icon=":material/search_off:")

# ── 공통 크기 (실시간, 작업 해상도 마스크로 계산) ───────────────────
for it in items:
    dims = crop_dims(it["mask"], it["orig"], shrink, feather)
    it["empty"] = dims is None            # 경계를 지나치게 줄여 시편 영역이 사라짐 (B-09)
    it["dims"] = dims or it["orig"]

size = (even_size(ss.cw, ss.ch) if ss.custom
        else optimal_size_from_dims([it["dims"] for it in items], margin, size_mode))
for it in items:
    it["out_name"] = output_name(it["stem"], fmt)
    it["upscaled"] = fit_scale(it["dims"], size, margin) > 1.02

empties = [it for it in items if it["empty"]]
if empties:
    st.warning(f"경계를 너무 많이 줄여 시편 영역이 사라진 사진 {len(empties)}장은 배경이 지워지지 않은 "
               "원본 그대로 저장됩니다. '경계 축소' 값을 낮춰 주세요: "
               + ", ".join(escape(it["stem"]) for it in empties), icon=":material/hide_image:")

# ── 원본 해상도 결과 (다운로드·저장할 때만 만든다, B-05) ──────────────
# 다운로드 콜백은 별도 스레드에서 실행되므로 지금의 설정값을 기본 인자로 묶어 둔다.
EXPORT_ARGS = (shrink, feather, size, margin, bg, fmt)


def export_files(snapshot=tuple(items), args=EXPORT_ARGS) -> dict[str, bytes]:
    return {it["out_name"]: render_full(it["data"], it["mask"], *args) for it in snapshot}


def export_one(it, args=EXPORT_ARGS):
    return lambda: render_full(it["data"], it["mask"], *args)


# ── 결과 요약 카드 ─────────────────────────────────────────────
n_ai = sum(it["method"] == METHOD_AI for it in items)
n_up = sum(it["upscaled"] for it in items)
with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("처리한 사진", f"{len(items)}장",
              f"AI {n_ai} · 로컬 {len(items) - n_ai}" + (f" · {scope}" if use_ai else ""),
              delta_color="off", delta_arrow="off")
    m2.metric("공통 출력 크기", f"{size[0]}×{size[1]}", "직접 지정" if ss.custom else f"{size_mode} 기준",
              delta_color="off", delta_arrow="off")
    m3.metric("저장 형식", fmt, bg_mode + " 배경" if not (fmt == "JPG" and bg[3] == 0) else "흰색 배경 (JPG)",
              delta_color="off", delta_arrow="off")
    m4.metric("확대된 사진", f"{n_up}장", "화질 저하 가능" if n_up else "원본 해상도 이내",
              delta_color="inverse" if n_up else "off", delta_arrow="off")

    # ── 저장 (PRD F-40~F-44) ──
    b1, b2 = st.columns([2, 1])
    b1.download_button("전체 ZIP 다운로드", lambda: build_zip(export_files()), icon=":material/download:",
                       file_name=f"specimens_{size[0]}x{size[1]}.zip", mime="application/zip",
                       width="stretch", type="primary", on_click="ignore")
    if b2.button("output 폴더에 저장", icon=":material/folder:", width="stretch"):
        try:
            with st.spinner("원본 해상도로 저장하는 중…"):
                out = save_to_folder(export_files(), config.OUTPUT_DIR)
            ss["saved_to"] = str(out)
            st.toast(f"{len(items)}장 저장 완료", icon="💾")
        except OSError as e:
            st.error(f"폴더에 저장하지 못했습니다: {e}")
    if ss.get("saved_to"):
        st.caption(f"💾 마지막 저장 위치: {ss['saved_to']}")


# ── 미리보기 (PRD F-34~F-36) ─────────────────────────────────
def preview(it, max_side: int):
    """저장될 결과와 같은 모습의 축소 미리보기 (JPG면 흰 배경, B-04)."""
    return render_preview(it["thumb"], it["mask"], it["orig"], shrink, feather, size, margin,
                          bg, fmt, max_side)


def method_badge(it):
    if it["method"] == METHOD_AI:
        st.badge("AI + GrabCut", icon=":material/auto_awesome:", color="green")
    elif "오류" in it["detail"]:
        st.badge("AI 오류 → 로컬", icon=":material/error:", color="red")
    else:
        st.badge("로컬", icon=":material/computer:", color="gray")
    if it["empty"]:
        st.badge("시편 영역 없음", icon=":material/hide_image:", color="orange")


tab_gallery, tab_compare = st.tabs(["🖼️ 결과 갤러리", "🔍 비교 보기"])

with tab_gallery:
    st.segmented_control("썸네일 크기", ["작게", "보통", "크게"], key="density",
                         on_change=keep_selection, args=("density",))
    n_cols = {"작게": 6, "보통": 4, "크게": 2}[ss.density]
    cols = st.columns(n_cols)
    for i, it in enumerate(items):
        with cols[i % len(cols)]:
            st.image(preview(it, 900 if n_cols <= 2 else 480), width="stretch")
            st.caption(escape(it["out_name"]) + (" · ⚠️ 확대됨" if it["upscaled"] else "")
                       + (" · ⚠️ 시편 영역 없음" if it["empty"] else ""))

with tab_compare:
    show_outline = st.toggle("원본에 테두리 표시", value=True)
    for it in items:
        with st.container(border=True):
            h1, h2 = st.columns([3, 1], vertical_alignment="center")
            with h1:
                st.markdown(f"**{escape(it['stem'])}** <small style='color:#6B7280'>· "
                            f"{it['orig'][0]}×{it['orig'][1]}</small>", unsafe_allow_html=True)
                method_badge(it)
            # key는 업로드 순번 기반 → 파일명이 어떻게 겹쳐도 중복되지 않음 (B-02)
            h2.download_button("다운로드", export_one(it), file_name=it["out_name"],
                               icon=":material/download:", key=f"dl_{it['idx']}", width="stretch",
                               on_click="ignore")
            st.caption(escape(it["detail"]))

            thumb = it["thumb"]
            left = thumb
            if show_outline:
                left = outline_overlay(thumb, scaled_mask(it["mask"], thumb.size, it["orig"][0],
                                                          shrink, feather))
            a, b = st.columns(2)
            a.image(left, caption="원본" + (" + 추출된 테두리" if show_outline else ""), width="stretch")
            b.image(preview(it, 800), caption=f"결과 · {size[0]}×{size[1]}", width="stretch")
