"""시편 테두리 추출과 마스크 생성.

처리 흐름 (PRD 6.1)
    ai_detect_outline → refine_with_grabcut → refine_mask → cutout

이 모듈은 Streamlit에 의존하지 않는다. 캐시(st.cache_data)는 app.py에서 씌운다.
비용이 큰 단계(ai_detect_outline, refine_with_grabcut)의 인자에 슬라이더 후처리 값을
넣으면 캐시가 깨지고 API를 다시 호출하게 되므로 넣지 말 것 (PRD N-02).
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from . import config
from .io_utils import load_image, resize_max

# 시편 범위 (사용자 선택). 테이프·프레임을 시편에 넣을지 프롬프트에 명시해 결과를 일관되게 만든다.
SCOPE_WITH_TAPE = "테이프 포함"
SCOPE_METAL_ONLY = "금속 노출부만"
SCOPES = (SCOPE_WITH_TAPE, SCOPE_METAL_ONLY)

_SCOPE_RULES = {
    SCOPE_WITH_TAPE: """\
- SPECIMEN EXTENT (mandatory): the specimen INCLUDES everything attached to it: masking tape,
  tape frames/borders, protective film, and labels or handwriting on that tape.
  Trace the OUTERMOST edge of the tape/frame, never the inner edge where the bare surface begins.
  Even if the tape and the bare surface look clearly different, they are ONE specimen.""",
    SCOPE_METAL_ONLY: """\
- SPECIMEN EXTENT (mandatory): trace ONLY the exposed bare surface (e.g. metal) of the specimen.
  EXCLUDE masking tape, tape frames/borders, protective film and labels on the tape.
  Follow the inner edge of the tape, where the bare surface begins.""",
}

_PROMPT_TEMPLATE = """You are a precise image segmentation assistant for laboratory/material test specimens.
Find the single main test specimen in the image and trace its boundary.

Return ONLY JSON in this exact schema:
{{
  "found": true,
  "bbox": [x_min, y_min, x_max, y_max],
  "polygon": [[x, y], [x, y], ...],
  "description": "short description of the specimen"
}}

Rules:
{scope_rule}
- All coordinates are normalized to 0..1 relative to image width (x) and height (y).
- "polygon" must follow that boundary closely, clockwise, 12 to 60 points.
  Include corners and notches (e.g. dog-bone / dumbbell shapes). Inner holes are NOT needed.
- Exclude background, table/workbench, rulers, loose paper labels, hands, shadows,
  reflections on the table and fixtures.
- If no specimen is visible, return {{"found": false}}.
"""


def build_prompt(scope: str = SCOPE_WITH_TAPE) -> str:
    if scope not in _SCOPE_RULES:
        raise ValueError(f"알 수 없는 시편 범위: {scope}")
    return _PROMPT_TEMPLATE.format(scope_rule=_SCOPE_RULES[scope])


PROMPT = build_prompt(SCOPE_WITH_TAPE)   # 하위 호환용 기본 프롬프트

METHOD_AI = "AI+GrabCut"
METHOD_LOCAL = "로컬"


# ──────────────────────────────────────────────────────────────
# AI 외곽선 추출 (PRD F-10, F-11)
# ──────────────────────────────────────────────────────────────
def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _parse_outline(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    data = json.loads(match.group(0) if match else (text or "{}"))
    if not isinstance(data, dict):
        raise ValueError("모델 응답이 JSON 객체가 아닙니다")
    return data


def ai_detect_outline(data: bytes, model: str, api_key: str | None = None,
                      client_factory: Callable | None = None,
                      scope: str = SCOPE_WITH_TAPE) -> dict:
    """OpenAI 비전 모델에게 시편 외곽선을 요청한다.

    scope: SCOPE_WITH_TAPE(테이프·프레임 포함, 기본) 또는 SCOPE_METAL_ONLY(금속 노출부만).

    Returns:
        {"found": bool, "bbox": [x0, y0, x1, y1], "polygon": [[x, y], ...], "description": str}
        좌표는 0~1로 정규화된 값.
    Raises:
        RuntimeError: API 키 없음. 그 밖의 API/파싱 오류는 그대로 전달된다.
    """
    key = api_key if api_key is not None else config.API_KEY
    if not key:
        raise RuntimeError("API 키가 없습니다")
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    client = client_factory(api_key=key)

    small, _ = resize_max(load_image(data), config.AI_MAX_SIDE)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": build_prompt(scope)},
            {"role": "user", "content": [
                {"type": "text", "text": "Trace the specimen boundary following the SPECIMEN EXTENT rule "
                                         "and return the JSON."},
                {"type": "image_url", "image_url": {"url": _to_data_url(small), "detail": "high"}},
            ]},
        ],
    )
    return _parse_outline(resp.choices[0].message.content)


# ──────────────────────────────────────────────────────────────
# GrabCut 초기 마스크
# ──────────────────────────────────────────────────────────────
def _clip01(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def init_mask_from_ai(shape: tuple[int, int], outline: dict) -> np.ndarray | None:
    """AI 외곽선으로 GrabCut 초기 마스크를 만든다. 쓸 수 없는 응답이면 None.

    다각형 깊은 내부 = FGD, 다각형 = PR_FGD, 다각형 주변 띠 = PR_BGD,
    bbox(+4%) 안 = PR_BGD, 그 밖 = BGD
    """
    h, w = shape
    if not outline or outline.get("found") is False:
        return None

    poly = outline.get("polygon") or []
    bbox = outline.get("bbox")
    try:
        pts = np.array([[_clip01(p[0]) * (w - 1), _clip01(p[1]) * (h - 1)] for p in poly], np.int32)
    except (TypeError, IndexError, ValueError):
        pts = np.empty((0, 2), np.int32)
    valid_bbox = isinstance(bbox, (list, tuple)) and len(bbox) == 4
    if len(pts) < 3 and not valid_bbox:
        return None

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    if valid_bbox:
        try:
            x0, y0, x1, y1 = (_clip01(v) for v in bbox)
        except (TypeError, ValueError):
            x0 = y0 = 0.0
            x1 = y1 = 1.0
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        pad = 0.04
        mask[max(0, int((y0 - pad) * h)):min(h, int((y1 + pad) * h)),
             max(0, int((x0 - pad) * w)):min(w, int((x1 + pad) * w))] = cv2.GC_PR_BGD

    if len(pts) >= 3:
        poly_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        if cv2.countNonZero(poly_mask) < 0.001 * h * w:   # 너무 작은 다각형은 무시
            return init_mask_from_ai(shape, {"bbox": bbox}) if valid_bbox else None
        k = max(3, int(min(h, w) * 0.03))
        kernel = np.ones((k, k), np.uint8)
        near = cv2.dilate(poly_mask, kernel)
        core = cv2.erode(poly_mask, kernel)
        mask[(near > 0) & (mask == cv2.GC_BGD)] = cv2.GC_PR_BGD
        mask[poly_mask > 0] = cv2.GC_PR_FGD
        mask[core > 0] = cv2.GC_FGD
    else:
        # bbox만 있을 때: bbox 안쪽 영역을 전경 후보로
        bx0, by0, bx1, by1 = (int(_clip01(bbox[0]) * w), int(_clip01(bbox[1]) * h),
                              int(_clip01(bbox[2]) * w), int(_clip01(bbox[3]) * h))
        mask[min(by0, by1):max(by0, by1), min(bx0, bx1):max(bx0, bx1)] = cv2.GC_PR_FGD
    return mask


def init_mask_local(shape: tuple[int, int]) -> np.ndarray:
    """AI 없이 쓰는 초기 마스크: 가장자리 8%는 배경, 가운데는 전경 후보."""
    h, w = shape
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mx, my = max(1, int(w * 0.08)), max(1, int(h * 0.08))
    mask[my:h - my, mx:w - mx] = cv2.GC_PR_FGD
    return mask


def _clean(mask: np.ndarray) -> np.ndarray:
    """가장 큰 덩어리만 남기고 내부 빈틈을 채운다 (PRD F-13)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(labels == biggest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


# ──────────────────────────────────────────────────────────────
# 마스크 생성 (PRD F-12~F-17)
# ──────────────────────────────────────────────────────────────
def describe_error(e: Exception) -> str:
    """API 오류를 사용자에게 보여 줄 문구로 바꾼다. 키 원문이나 일부가 남지 않게 한다 (N-01)."""
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    if name == "AuthenticationError" or status == 401:
        return "API 키가 올바르지 않습니다 (401). .env의 키를 확인해 주세요"
    if name == "PermissionDeniedError" or status == 403:
        return "이 API 키로는 모델을 사용할 권한이 없습니다 (403)"
    msg = str(e)
    if config.API_KEY:
        msg = msg.replace(config.API_KEY, "***")
    msg = re.sub(r"sk-[A-Za-z0-9_\-*]{4,}", "sk-***", msg)     # OpenAI가 가린 형태(sk-ab***cd)도 제거
    return f"{name}: {msg[:150]}"


def segment_work(data: bytes, use_ai: bool, model: str, iters: int,
                 outline_fn: Callable[[bytes, str], dict] | None = None,
                 scope: str = SCOPE_WITH_TAPE,
                 ) -> tuple[np.ndarray, tuple[int, int], str, str]:
    """AI 외곽선(또는 로컬 초기값)으로 GrabCut을 돌려 작업 해상도 마스크를 만든다.

    작업 해상도(긴 변 ≤ WORK_MAX_SIDE) 그대로 돌려주므로 캐시 메모리가 작다 (B-06).
    원본 크기가 필요하면 mask_to_size()로 키운다.

    Args:
        outline_fn: (data, model) -> outline dict. 기본값은 scope를 적용한 ai_detect_outline.
                    app.py에서는 캐시된 버전을, 테스트에서는 가짜 함수를 넣는다.
        scope: 시편 범위 (outline_fn을 주지 않았을 때만 사용).
    Returns:
        (작업 해상도 uint8 마스크 0/255, 원본 크기 (W, H), 처리 방식 METHOD_AI|METHOD_LOCAL, 상세 설명)
        AI 단계에서 오류가 나도 예외를 던지지 않고 로컬 방식으로 처리한다 (PRD F-15, N-05).
    """
    img = load_image(data)
    work, scale = resize_max(img, config.WORK_MAX_SIDE)
    arr = cv2.cvtColor(np.array(work), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]

    init, method, detail = None, METHOD_LOCAL, "AI 사용 안 함"
    if use_ai:
        try:
            fn = outline_fn or (lambda d, m: ai_detect_outline(d, m, scope=scope))
            outline = fn(data, model)
            init = init_mask_from_ai((h, w), outline)
            if init is not None:
                method, detail = METHOD_AI, f"{model} 외곽선 + GrabCut"
            else:
                detail = "AI가 시편을 찾지 못해 로컬 방식으로 처리"
        except Exception as e:
            detail = f"AI 오류로 로컬 방식으로 처리 ({describe_error(e)})"
    if init is None:
        init = init_mask_local((h, w))

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    gc = init.copy()
    try:
        cv2.grabCut(arr, gc, None, bgd, fgd, max(1, int(iters)), cv2.GC_INIT_WITH_MASK)
        m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        # 전경/배경 샘플이 한쪽뿐이면 GrabCut이 실패한다 → 초기 마스크를 그대로 사용
        m = np.where((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        detail += " · GrabCut 실패, 초기 마스크 사용"
    return _clean(m), img.size, method, detail


def mask_to_size(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """마스크를 (W, H) 크기로 바꾼다. 확대 후 다시 이진화해 경계를 또렷하게 유지한다."""
    if mask.shape[::-1] == tuple(size):
        return mask
    m = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_LINEAR)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def refine_with_grabcut(data: bytes, use_ai: bool, model: str, iters: int,
                        outline_fn: Callable[[bytes, str], dict] | None = None,
                        scope: str = SCOPE_WITH_TAPE) -> tuple[np.ndarray, str, str]:
    """segment_work의 결과를 원본 크기 마스크로 돌려준다.

    Returns:
        (원본 크기의 uint8 마스크 0/255, 처리 방식, 상세 설명)
    """
    m, orig_size, method, detail = segment_work(data, use_ai, model, iters, outline_fn, scope)
    m = mask_to_size(m, orig_size)
    return m, method, detail


# ──────────────────────────────────────────────────────────────
# 실시간 후처리 (PRD F-14, F-31, F-32, F-35)
# ──────────────────────────────────────────────────────────────
def refine_mask(mask: np.ndarray, shrink: int, feather: int) -> np.ndarray:
    """경계 축소(+)/확장(−)과 가장자리 부드럽게를 적용한다."""
    m = mask
    if shrink:
        k = np.ones((2 * abs(shrink) + 1,) * 2, np.uint8)
        m = cv2.erode(m, k) if shrink > 0 else cv2.dilate(m, k)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def cutout(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """마스크를 알파 채널로 써서 시편 영역만 잘라낸 RGBA 이미지를 만든다.

    마스크가 비어 있으면 원본 전체를 반환한다 (축소를 크게 했을 때 0px 이미지 방지).
    """
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask))
    bbox = Image.fromarray(np.where(mask > 10, 255, 0).astype(np.uint8)).getbbox()
    return rgba.crop(bbox) if bbox else img.convert("RGBA")


def outline_overlay(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """원본 위에 추출된 테두리를 빨간 선으로 그린 비교용 이미지."""
    arr = np.array(img.convert("RGB")).copy()
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, (255, 40, 80), max(2, int(max(img.size) / 300)))
    return Image.fromarray(arr)
