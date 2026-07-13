"""Local OMR demo server: proxies inference to Colab and renders Verovio scores."""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STATIC = Path(__file__).resolve().parent / "static"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from primus_semantic_render import render_semantic_to_image  # noqa: E402

app = FastAPI(title="OMR Demo", docs_url=None, redoc_url=None)

DEFAULT_COLAB_URL = os.environ.get("COLAB_INFER_URL", "").rstrip("/")
REQUEST_TIMEOUT_S = float(os.environ.get("COLAB_TIMEOUT_S", "180"))


def _colab_base(override: str | None = None) -> str:
    base = (override or DEFAULT_COLAB_URL or "").strip().rstrip("/")
    if not base:
        raise HTTPException(
            status_code=400,
            detail="Set COLAB_INFER_URL env var or pass colab_url in the form.",
        )
    return base


def _image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bytes_to_preview_b64(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return _image_to_b64_png(img)


def _render_semantic(semantic: str) -> str:
    text = " ".join((semantic or "").split())
    if not text:
        raise HTTPException(status_code=400, detail="Semantic string is empty.")
    try:
        rendered = render_semantic_to_image(text)
    except Exception as exc:  # Verovio/Playwright failures
        raise HTTPException(
            status_code=500,
            detail=f"Render failed: {type(exc).__name__}: {exc}",
        ) from exc
    return _image_to_b64_png(rendered)


async def _call_colab_infer(colab_url: str, image_bytes: bytes, filename: str) -> str:
    headers = {"ngrok-skip-browser-warning": "true"}
    files = {"image": (filename or "score.png", image_bytes, "application/octet-stream")}
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, follow_redirects=True) as client:
            response = await client.post(f"{colab_url}/infer", files=files, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Colab server at {colab_url}: {exc}",
        ) from exc

    if response.status_code >= 400:
        detail = response.text[:500]
        raise HTTPException(
            status_code=502,
            detail=f"Colab /infer returned {response.status_code}: {detail}",
        )

    payload = response.json()
    semantic = payload.get("semantic")
    if not isinstance(semantic, str) or not semantic.strip():
        raise HTTPException(status_code=502, detail=f"Colab response missing semantic: {payload}")
    return " ".join(semantic.split())


@app.get("/api/health")
async def health(colab_url: str | None = None):
    base = None
    colab_ok = None
    colab_error = None
    try:
        base = _colab_base(colab_url)
    except HTTPException:
        base = None

    if base:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(
                    f"{base}/health",
                    headers={"ngrok-skip-browser-warning": "true"},
                )
            colab_ok = resp.status_code == 200
            if not colab_ok:
                colab_error = f"status {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            colab_ok = False
            colab_error = str(exc)

    return {
        "ok": True,
        "colab_url": base,
        "colab_ok": colab_ok,
        "colab_error": colab_error,
        "render": "verovio+playwright",
    }


@app.post("/api/transcribe")
async def transcribe(
    image: UploadFile = File(...),
    colab_url: str | None = Form(None),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    base = _colab_base(colab_url)
    semantic = await _call_colab_infer(base, image_bytes, image.filename or "score.png")
    rendered_b64 = _render_semantic(semantic)
    input_b64 = _bytes_to_preview_b64(image_bytes)

    return {
        "semantic": semantic,
        "input_image_b64": input_b64,
        "rendered_image_b64": rendered_b64,
        "colab_url": base,
    }


@app.post("/api/render")
async def render_only(semantic: str = Form(...)):
    rendered_b64 = _render_semantic(semantic)
    return {
        "semantic": " ".join(semantic.split()),
        "rendered_image_b64": rendered_b64,
    }


@app.get("/")
async def index():
    index_path = STATIC / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail=f"Missing frontend at {index_path}")
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "7860")),
        reload=False,
    )
