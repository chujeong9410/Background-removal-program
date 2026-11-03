"""미리보기와 최종 결과 만들기.

실시간 미리보기는 축소본으로 만들고(빠름), 원본 해상도 결과는 다운로드·저장할 때만
만든다 (B-05). 두 경로 모두 같은 공통 크기 size를 쓰므로 저장 결과의 크기는
미리보기에 표시된 크기와 항상 같다.

    work_mask  : segment_work()의 작업 해상도 마스크 (캐시 대상)
    orig_size  : 원본 이미지 크기 (W, H)
"""

from __future__ import annotations

from PIL import Image

from .io_utils import encode, load_image
from .layout import checker, place_on_canvas
from .segment import cutout, mask_to_size, refine_mask


def _scaled_params(shrink: int, feather: int, ratio: float) -> tuple[int, float]:
    """원본 해상도 기준의 경계 조정 값을 다른 해상도에 맞게 바꾼다."""
    # 축소본에서 1px 미만이 되는 조정은 실제로도 눈에 보이지 않으므로 그대로 0으로 둔다
    return round(shrink * ratio), feather * ratio


def crop_dims(work_mask, orig_size: tuple[int, int], shrink: int, feather: int
              ) -> tuple[int, int] | None:
    """경계 조정 후 시편 영역의 크기를 원본 해상도 기준으로 추정한다.

    Returns:
        (w, h). 시편 영역이 모두 사라졌으면 None (B-09).
    """
    ratio = work_mask.shape[1] / orig_size[0]
    s, f = _scaled_params(shrink, feather, ratio)
    m = refine_mask(work_mask, s, f)
    bbox = Image.fromarray((m > 10).astype("uint8") * 255).getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return (max(1, round((x1 - x0) / ratio)), max(1, round((y1 - y0) / ratio)))


def scaled_mask(work_mask, target_size: tuple[int, int], orig_width: int,
                shrink: int, feather: int):
    """작업 마스크를 target_size 해상도로 바꾸고 경계 조정을 그 해상도에 맞게 적용한다."""
    s, f = _scaled_params(shrink, feather, target_size[0] / orig_width)
    return refine_mask(mask_to_size(work_mask, target_size), s, f)


def compose(img: Image.Image, work_mask, shrink: int, feather: int, size: tuple[int, int],
            margin: float, bg: tuple[int, int, int, int], orig_width: int) -> Image.Image:
    """img 해상도에서 배경 제거 → 캔버스 배치. img는 원본이거나 축소본."""
    m = scaled_mask(work_mask, img.size, orig_width, shrink, feather)
    return place_on_canvas(cutout(img, m), size, margin, bg)


def preview_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    k = min(1.0, max_side / max(size))
    return max(1, round(size[0] * k)), max(1, round(size[1] * k))


def render_preview(thumb: Image.Image, work_mask, orig_size: tuple[int, int], shrink: int,
                   feather: int, size: tuple[int, int], margin: float,
                   bg: tuple[int, int, int, int], fmt: str, max_side: int = 640) -> Image.Image:
    """축소본으로 만든 미리보기. 저장될 결과와 같은 모습으로 보여 준다 (B-04).

    - JPG: 투명 영역을 흰색으로 합성해서 보여 준다 (실제 저장 결과와 같음)
    - PNG + 투명 배경: 체크무늬 위에 보여 준다
    """
    canvas = compose(thumb, work_mask, shrink, feather, preview_size(size, max_side),
                     margin, bg, orig_size[0])
    if fmt.upper() == "JPG":
        flat = Image.new("RGB", canvas.size, (255, 255, 255))
        flat.paste(canvas, mask=canvas.getchannel("A"))
        return flat
    return checker(canvas) if bg[3] == 0 else canvas.convert("RGB")


def render_full(data: bytes, work_mask, shrink: int, feather: int, size: tuple[int, int],
                margin: float, bg: tuple[int, int, int, int], fmt: str) -> bytes:
    """원본 해상도에서 최종 결과를 만들어 인코딩한다 (다운로드·저장 시에만 호출)."""
    img = load_image(data)
    return encode(compose(img, work_mask, shrink, feather, size, margin, bg, img.width), fmt)
