from __future__ import annotations

import os
import sys

from src.core.registry import get_models_for_input_type, get_xai_for
from src.core.inference import run_prediction, run_explanations


SAMPLE_IMAGE = os.path.join("samples", "sample.jpg")
SAMPLE_AUDIO = os.path.join("samples", "sample.wav")


def _run_sample(path: str, input_type: str):
    if not os.path.exists(path):
        print(f"SKIP: {path} not found.")
        return True
    models = get_models_for_input_type(input_type)
    if not models:
        print(f"FAIL: no models for {input_type}.")
        return False
    model = models[0]
    sample = model.preprocess(path)
    pred = run_prediction(model, sample)
    xai = get_xai_for(input_type, model.name)
    if not xai:
        print(f"FAIL: no XAI for {input_type}.")
        return False
    run_explanations(model, sample, pred, [xai[0]])
    print(f"OK: {input_type} sample processed with {model.name} + {xai[0].name}.")
    return True


def main():
    ok = True
    ok = _run_sample(SAMPLE_IMAGE, "image") and ok
    ok = _run_sample(SAMPLE_AUDIO, "audio") and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
