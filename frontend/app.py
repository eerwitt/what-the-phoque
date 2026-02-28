import datetime
import io
import logging
import os
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEEDBACK_DATASET = "eerwitt/what-the-phoque-feedback"
HF_TOKEN = os.environ.get("HF_TOKEN")

app = FastAPI()
hf_api = HfApi(token=HF_TOKEN)


class FeedbackPayload(BaseModel):
    session_id: str
    prompt: str
    response: str
    rating: int = Field(..., ge=1, le=5)
    toxicity_types: List[str] = []
    notes: Optional[str] = ""
    model_id: str = "eerwitt/what-the-phoque-onnx"


@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackPayload):
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="HF_TOKEN not configured")

    path_in_repo = f"data/{payload.session_id}.parquet"

    # Try to fetch and append to the existing session parquet
    df_existing = None
    try:
        local_path = hf_hub_download(
            repo_id=FEEDBACK_DATASET,
            filename=path_in_repo,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        df_existing = pd.read_parquet(local_path)
    except (EntryNotFoundError, RepositoryNotFoundError):
        pass
    except Exception as exc:
        logger.warning("Could not fetch existing parquet: %s", exc)

    new_row = pd.DataFrame(
        [
            {
                "session_id": payload.session_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "prompt": payload.prompt,
                "response": payload.response,
                "rating": payload.rating,
                "toxicity_types": payload.toxicity_types,
                "notes": payload.notes or "",
                "model_id": payload.model_id,
            }
        ]
    )

    df = (
        pd.concat([df_existing, new_row], ignore_index=True)
        if df_existing is not None
        else new_row
    )

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    try:
        hf_api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=path_in_repo,
            repo_id=FEEDBACK_DATASET,
            repo_type="dataset",
            commit_message=f"Add feedback for session {payload.session_id[:8]}",
        )
    except Exception as exc:
        logger.error("Failed to upload feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save feedback") from exc

    return {"status": "ok", "rows": len(df)}


# Serve the compiled React app for all other routes (must be last)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
