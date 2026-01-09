# Unified XAI Lab Technical Report

## Introduction
This project integrates two explainable AI workflows (audio deepfake detection and chest X-ray classification) into a single, unified interface. The goal is to support multimodal inputs, multiple pretrained models, and comparable explanation methods in a consistent user experience for analysis and learning.

## Related Work
The system is derived from two prior codebases: an audio deepfake detector using CNN-based classifiers with XAI overlays, and a chest X-ray classifier with Grad-CAM visualizations. We unify their pipelines and expose a single interface for model selection and explanation comparison across modalities.

## Methodology
We standardize the pipeline around shared abstractions for input samples, prediction outputs, and explanation outputs. Each model wrapper implements preprocessing, prediction, and tensor conversion. XAI wrappers receive model and sample context and return renderable artifacts (heatmaps, overlays, or originals). The UI uses dynamic filtering to ensure method compatibility with input type and offers single and comparison workflows.

## Experiments
We exercised both the image and audio pipelines using local sample files. For images, we tested AlexNet and DenseNet with Grad-CAM, LIME, and SHAP; for audio, we tested AudioCNN with LIME and SHAP on mel spectrograms. We validated that the comparison tab renders multiple explanation images side-by-side and that each output is annotated with the selected model, method, and prediction summary.

## Results
The application successfully supports:
- Image and audio inputs with automatic type detection.
- Multiple models per modality with compatible XAI methods.
- A comparison view that displays prediction summaries and method-labeled explanation outputs.
- Single-explanation view with interactive overlay and zoom controls.

## Discussion
Predictions are produced by randomly initialized model weights in this baseline implementation. This is sufficient to validate the interface, but not the predictive quality. Replacing with trained weights is required for meaningful clinical or forensic interpretation. The interface design prioritizes clarity, method labeling, and compatibility filtering to avoid user confusion.

## Conclusion
We delivered a unified, explainable AI interface that combines two modalities, multiple models, and multiple XAI techniques. The interface supports comparison and interactive exploration while keeping the codebase modular and extensible.

## Reproducibility
- Create a virtual environment and install dependencies from `requirements.txt`.
- Run `python app.py` and open the Gradio link.
- Use the provided samples or upload your own `.wav` and `.png/.jpg` files.
- Note: WAV PCM (e.g., 16-bit) is recommended for stable audio preprocessing.

## Generative AI Usage Statement
An AI coding assistant was used to help draft documentation and apply targeted UI edits. The final implementation and verification were performed by the project authors.

## References
- Original audio deepfake detection repository (FoR dataset pipeline).
- Original lung cancer detection repository (CheXpert pipeline).
- Gradio documentation for UI components.
