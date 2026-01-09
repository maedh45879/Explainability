# Compliance Memo - Unified XAI Lab

## Summary
This memo maps each rubric requirement from the project guidelines to evidence in the repository. Screenshot placeholders are included where visual proof is needed.

## Requirement Mapping

| Requirement | Evidence | Screenshot Placeholder |
| --- | --- | --- |
| Unified interface for audio (.wav) and image (.png/.jpg) inputs | `app.py`, `src/ui/pages.py` | `screenshots/ui_home.png` |
| Multiple pretrained models per modality | `src/core/registry.py`, `src/models/image_models.py`, `src/models/audio_models.py` | `screenshots/model_dropdown.png` |
| XAI methods (LIME, Grad-CAM, SHAP) integrated | `src/xai/`, `src/core/registry.py` | `screenshots/xai_methods.png` |
| Automatic filtering of XAI methods by input type | `src/core/registry.py`, `src/ui/pages.py` | `screenshots/filtered_methods.png` |
| Basic GUI flow: select dataset/model/method and show explanation | `src/ui/pages.py` (Single Explanation tab) | `screenshots/single_explanation.png` |
| Comparison tab with multiple explainability outputs | `src/ui/pages.py` (Compare tab) | `screenshots/compare_tab.png` |
| Comparison tab shows classification results alongside explanations | `src/ui/pages.py` (prediction summary + captions) | `screenshots/compare_predictions.png` |
| Method labels clearly indicated on each visualization | `src/ui/pages.py` (gallery captions) | `screenshots/method_labels.png` |
| User-friendly interface and clarity | `src/ui/pages.py`, `README.md` | `screenshots/ui_clarity.png` |
| Documentation: README with setup + usage | `README.md` | `screenshots/readme.png` |
| Documentation: short technical report | `report.md` | N/A |
| Demo readiness (audio + image flow) | `scripts/self_check.py`, manual run notes | `screenshots/demo_flow.png` |
| Generative AI usage declared | `README.md`, `report.md` | N/A |

## Notes
- Prediction outputs are displayed alongside explanations in the Compare tab via a structured summary and per-image captions.
- Screenshots should be captured during the final demo run and saved using the placeholders listed above.
