# Local OMR demo (localhost + Colab)

One-time demo: upload a monophonic score image in the browser, run inference on a Colab GPU with your fine-tuned adapter [`NathanZ721/omr-monophonic`](https://huggingface.co/NathanZ721/omr-monophonic), then render a clean digital score locally with Verovio.

## Architecture

1. **Colab** — loads Qwen2-VL-2B + LoRA, exposes `POST /infer` through ngrok  
2. **Local FastAPI** (`demo/server.py`) — receives the upload, calls Colab, renders PrIMuS → PNG  
3. **Browser** — upload UI at `http://127.0.0.1:7860`

## 1. Start the Colab server

1. Open [`notebooks/demo_omr_colab_server.ipynb`](../notebooks/demo_omr_colab_server.ipynb) in Google Colab.
2. Use a **GPU** runtime (T4 is enough).
3. Paste your [ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) into cell 4 (`NGROK_AUTHTOKEN`).
4. Run all cells.
5. Copy the printed **Public URL** (for example `https://xxxx.ngrok-free.app`).

Keep the Colab runtime alive while you demo.

## 2. Install local dependencies

From the repo root (Windows — use `py -3 -m`, since `pip` / `playwright` may not be on PATH):

```powershell
py -3 -m pip install fastapi uvicorn python-multipart httpx verovio pillow playwright
py -3 -m playwright install chromium
```

## 3. Run the local app

```powershell
cd C:\alpha_ai\small-language-model
$env:COLAB_INFER_URL = "https://YOUR-NGROK-URL"
py -3 -m uvicorn demo.server:app --host 127.0.0.1 --port 7860
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860).

You can also paste the Colab URL in the page’s **Colab base URL** field (it is saved in localStorage).

## 4. Demo flow

1. Click **Check Colab** to verify `/health`.
2. Upload or drag a score image.
3. Click **Transcribe**.
4. Review the predicted PrIMuS semantic string and the rendered clean digital image.
5. Optionally edit the semantic text and click **Re-render** (local only; no Colab call).

## API

| Endpoint | Role |
| --- | --- |
| `GET /api/health` | Local status + optional Colab reachability |
| `POST /api/transcribe` | multipart `image` (+ optional `colab_url`) → semantic + PNGs |
| `POST /api/render` | form `semantic` → rendered PNG only |

Colab:

| Endpoint | Role |
| --- | --- |
| `GET /health` | Model loaded |
| `POST /infer` | multipart `image` → `{ "semantic": "..." }` |

## Notes

- Rendering uses [`scripts/primus_semantic_render.py`](../scripts/primus_semantic_render.py) (Verovio + Playwright).
- Inference matches the golden-eval prompts and greedy decoding (`max_new_tokens=256`).
- This is a local/Colab demo only — not production-hardened.
