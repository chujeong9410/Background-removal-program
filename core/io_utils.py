"""이미지 입출력: 로드(EXIF 회전 보정, RGB 변환)와 PNG/JPG 인코딩."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def image_hash(data: bytes) -> str:
    """캐시 키로 쓰는 이미지 바이트의 SHA-256 해시 (PRD N-02)."""
    return hashlib.sha256(data).hexdigest()


def load_image(data: bytes) -> Image.Image:
    """업로드된 바이트를 RGB 이미지로 읽는다. 휴대폰 사진의 EXIF 회전도 보정한다 (PRD F-05)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode.startswith("I;16") or img.mode in ("I", "F"):
        img = _to_8bit(img)
    return img.convert("RGB")


def _to_8bit(img: Image.Image) -> Image.Image:
    """16비트·32비트 정수·실수 이미지를 0~255로 맞춘다 (B-03).

    convert("RGB")는 값을 255에서 잘라 버려 전체가 흰색이 되므로, 실제 값 범위
    (0.5~99.5 백분위수)를 기준으로 늘려서 대비를 살린다.
    """
    arr = np.asarray(img, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return Image.fromarray(np.zeros(arr.shape, np.uint8))
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return Image.fromarray(np.full(arr.shape, 128, np.uint8))
    out = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo) * 255, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def resize_max(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """긴 변이 max_side를 넘으면 비율을 유지한 채 축소한다. (축소된 이미지, 배율)을 반환."""
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img, scale


def make_thumbnail(img: Image.Image, max_side: int = 320) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    return thumb


def unique_stems(names: list[str]) -> list[str]:
    """같은 이름이 여러 번 올라와도 결과 파일이 덮어써지지 않게 _2, _3을 붙인다 (B-02).

    - 원래 이름은 먼저 등장한 파일이 그대로 쓴다. 'a_2.jpg' 같은 실제 파일명도 보호한다.
    - Windows는 대소문자를 구분하지 않으므로 'A.jpg'와 'a.jpg'도 겹치는 것으로 본다.
    """
    stems = [Path(n).stem or "image" for n in names]
    out: list[str | None] = [None] * len(stems)
    used: set[str] = set()
    for i, s in enumerate(stems):          # 1차: 원래 이름 그대로 쓸 수 있는 파일
        if s.casefold() not in used:
            out[i] = s
            used.add(s.casefold())
    for i, s in enumerate(stems):          # 2차: 겹친 파일은 비어 있는 번호를 찾는다
        if out[i] is None:
            n = 2
            while f"{s}_{n}".casefold() in used:
                n += 1
            out[i] = f"{s}_{n}"
            used.add(out[i].casefold())
    return out


def output_name(stem: str, fmt: str) -> str:
    """PRD F-44 파일명 규칙: {원본 이름}_nobg.{확장자}"""
    return f"{stem}_nobg.{'png' if fmt.upper() == 'PNG' else 'jpg'}"


def build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def save_to_folder(files: dict[str, bytes], root: Path) -> Path:
    """root/YYYYMMDD_HHMMSS/ 폴더에 저장하고 폴더 경로를 반환한다 (PRD F-42)."""
    out = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    i = 1
    while out.exists():
        i += 1
        out = root / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i}")
    out.mkdir(parents=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    return out


def encode(img: Image.Image, fmt: str) -> bytes:
    """결과 이미지를 PNG(투명 유지) 또는 JPG(흰색 배경 위에 합성)로 인코딩한다 (PRD F-43)."""
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG", optimize=True)
    else:
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        flat.save(buf, format="JPEG", quality=95)
    return buf.getvalue()
