# Unified XAI Lab

Unified Explainable AI (XAI) web app for:
- Deepfake Audio Detection (wav)
- Lung Cancer Detection on chest X-rays (png/jpg)

The app auto-detects file type, filters compatible models/XAI methods, and compares explanations side-by-side.

## Team
- TD group: TBD
- Members: TBD

## Project Deliverables
- Technical report: `report.md`
- Compliance memo: `compliance_memo.md`

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Usage
1) Open the local Gradio link shown in the terminal.
2) Upload a `.wav` audio or `.png/.jpg` image.
3) Pick a model (choices depend on input type).
4) Select one or more XAI methods.
5) Use the controls to adjust overlays, opacity, and zoom.

## Supported XAI Methods
- Grad-CAM (image only)
- LIME (image + audio via mel spectrogram)
- SHAP (image + audio via mel spectrogram)
- Integrated Gradients (image only)

## Interactive Controls
- Image: overlay toggle, opacity slider, view mode (original/heatmap/overlay), zoom crop.
- Audio: waveform + spectrogram display, time window slider.

## Limitations
- Models use randomly initialized weights by default. Replace with trained weights for meaningful predictions.
- SHAP is slow for large inputs; this app uses small sample sizes.
- LIME/SHAP explanations are approximations and may be noisy.

## Troubleshooting
- Audio must be WAV PCM (e.g., 16-bit) for consistent loading; compressed formats may fail preprocessing.
- If dropdowns appear empty after switching inputs, re-upload the file to refresh the filtered model/XAI list.

## Model/XAI Registry
Add new models or methods by editing:
- `src/core/registry.py`
- `src/models/*.py`
- `src/xai/*.py`

### Add a new model
1) Create a wrapper in `src/models/` inheriting `BaseModelWrapper`.
2) Implement `preprocess`, `predict`, and `tensor_from_numpy`.
3) Register the wrapper in `src/core/registry.py`.

### Add a new XAI method
1) Create a wrapper in `src/xai/` inheriting `BaseXAIWrapper`.
2) Implement `explain` and set `compatible_input_types`.
3) Register it in `src/core/registry.py`.

## Self-check
```bash
python scripts/self_check.py
```
If `samples/sample.jpg` or `samples/sample.wav` do not exist, the self-check will skip them.

## Generative AI Usage Statement
This project structure and boilerplate code were assisted by an AI coding tool. The final implementation and verification are the user's responsibility.
