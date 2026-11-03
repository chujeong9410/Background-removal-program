"""공통 출력 크기 계산과 캔버스 배치.

공통 크기 계산식 (PRD 4.3)
    r      = median(너비ᵢ / 높이ᵢ)
    fit_hᵢ = min(높이ᵢ, 너비ᵢ / r)
    H      = {median | min | max}(fit_h) × (1 + 2×여백)
    W      = H × r      (짝수 픽셀로 반올림)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

SIZE_MODES = ("중앙값", "최소", "최대")  # PRD F-22
_AGG = {"중앙값": np.median, "최소": np.min, "최대": np.max}


def _even(v: float) -> int:
    return max(2, int(round(v / 2) * 2))


def even_size(w: float, h: float) -> tuple[int, int]:
    """크기를 짝수 픽셀로 맞춘다 (직접 지정 크기에도 적용, B-07)."""
    return _even(w), _even(h)


def optimal_size_from_dims(dims: list[tuple[int, int]], margin: float,
                           mode: str = "중앙값") -> tuple[int, int]:
    """시편 크기 목록 [(w, h), ...]로 공통 출력 크기 (W, H)를 계산한다 (PRD F-20~F-22)."""
    if not dims:
        return (0, 0)
    if mode not in _AGG:
        raise ValueError(f"알 수 없는 크기 기준: {mode}")
    ratios = np.array([w / h for w, h in dims], float)
    r = float(np.median(ratios))
    fit_h = np.array([min(h, w / r) for w, h in dims], float)
    h = float(_AGG[mode](fit_h)) * (1 + 2 * margin)
    return _even(h * r), _even(h)


def optimal_size(crops: list[Image.Image], margin: float, mode: str = "중앙값") -> tuple[int, int]:
    """잘라낸 시편 이미지들로 공통 출력 크기 (W, H)를 계산한다."""
    return optimal_size_from_dims([c.size for c in crops], margin, mode)


def place_on_canvas(crop: Image.Image, size: tuple[int, int], margin: float,
                    bg: tuple[int, int, int, int]) -> Image.Image:
    """시편을 비율을 유지한 채 size 캔버스 중앙에 배치한다 (PRD F-23, F-24, F-33)."""
    W, H = size
    inner_w, inner_h = W / (1 + 2 * margin), H / (1 + 2 * margin)
    s = min(inner_w / crop.width, inner_h / crop.height)
    nw, nh = max(1, round(crop.width * s)), max(1, round(crop.height * s))
    resized = crop.convert("RGBA").resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), tuple(bg))
    canvas.alpha_composite(resized, ((W - nw) // 2, (H - nh) // 2))
    return canvas


def fit_scale(crop_wh: tuple[int, int], size: tuple[int, int], margin: float) -> float:
    """크기 crop_wh인 시편을 size 캔버스에 넣을 때의 배율. 1보다 크면 확대된 것."""
    W, H = size
    return min(W / (1 + 2 * margin) / crop_wh[0], H / (1 + 2 * margin) / crop_wh[1])


def upscale_factor(crop: Image.Image, size: tuple[int, int], margin: float) -> float:
    """place_on_canvas에서 적용되는 배율. 1보다 크면 확대된 것."""
    return fit_scale(crop.size, size, margin)


def checker(img: Image.Image) -> Image.Image:
    """투명 영역이 보이도록 체크무늬 배경 위에 합성한 미리보기 (PRD F-36)."""
    w, h = img.size
    c = max(8, min(w, h) // 30)
    yy, xx = np.mgrid[0:h, 0:w]
    pat = np.where(((xx // c) + (yy // c)) % 2 == 0, 235, 205).astype(np.uint8)
    bg = Image.fromarray(np.stack([pat] * 3, -1)).convert("RGBA")
    bg.alpha_composite(img.convert("RGBA"))
    return bg.convert("RGB")
