"""
evaluation.py
-------------
GET /api/evaluation — Return stored model evaluation results.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, Request

from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])

_RESULTS_FILENAME = "evaluation_results.json"


@router.get("/evaluation", summary="Stored model evaluation results")
async def get_evaluation(request: Request) -> Dict[str, Any]:
    """
    Return previously computed evaluation metrics from disk.

    The results file is expected at ``{MODELS_PATH}/evaluation_results.json``.

    * If the file exists, its contents are returned with ``status: 'ok'``.
    * If the file is absent, ``status: 'not_run'`` is returned together with
      instructions for how to generate the results.
    * If the file exists but cannot be parsed, ``status: 'error'`` is returned.
    """
    settings = get_settings()
    results_path = os.path.join(settings.MODELS_PATH, _RESULTS_FILENAME)

    if not os.path.isfile(results_path):
        return {
            "status": "not_run",
            "message": (
                "Evaluation results have not been generated yet. "
                "Run the evaluation notebook or script at "
                "`notebooks/04_evaluation.ipynb` (or `scripts/evaluate.py`) "
                "to produce the results file, then restart the API."
            ),
            "expected_path": results_path,
            "results": None,
        }

    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("evaluation: failed to parse %s: %s", results_path, exc)
        return {
            "status": "error",
            "message": f"Evaluation results file exists but could not be parsed: {exc}",
            "expected_path": results_path,
            "results": None,
        }
    except OSError as exc:
        logger.error("evaluation: cannot read %s: %s", results_path, exc)
        return {
            "status": "error",
            "message": f"Could not read evaluation results file: {exc}",
            "expected_path": results_path,
            "results": None,
        }

    return {
        "status": "ok",
        "message": "Evaluation results loaded successfully.",
        "path": results_path,
        "results": data,
    }
