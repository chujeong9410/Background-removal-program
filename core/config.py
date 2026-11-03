"""환경 설정 로드 (.env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# OPENAI_API_KEY 또는 OPEN_API_KEY 둘 다 인식
API_KEY: str | None = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY") or None
MODEL: str = os.getenv("OPENAI_MODEL", "gpt-6-astra")

OUTPUT_DIR: Path = BASE_DIR / "output"

AI_MAX_SIDE = 1024     # 모델에 보낼 이미지의 최대 변 길이 (PRD N-03)
WORK_MAX_SIDE = 1600   # GrabCut 작업 해상도의 최대 변 길이 (PRD N-03)

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]  # PRD F-04


def has_api_key() -> bool:
    return bool(API_KEY)
