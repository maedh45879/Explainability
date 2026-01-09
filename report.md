# Unified XAI Lab – Unified Explainable AI Interface for Image and Audio Models

## 1. Introduction
Explainable AI (XAI) aims to make model decisions understandable to humans by revealing which input features influence predictions. This is particularly important for high-stakes domains where trust, accountability, and model debugging are required. Image and audio models pose different interpretability challenges, and users often need to compare methods across both modalities. The objective of this project is to provide a unified, interactive interface that supports image and audio inputs, runs multiple models, and visualizes multiple explanation techniques in a consistent workflow. The scope covers a Gradio-based application for demonstrations of XAI methods on images and audio, including comparison and basic interactive controls.

## 2. Project Objectives
The project targets the following objectives:
- Main functional objectives: provide a single UI for image and audio inputs, allow model selection, and display predictions alongside explanations.
- Explainability objectives: integrate multiple XAI techniques and present their outputs in a clear, comparable format.
- User-oriented objectives: minimize friction via automatic input-type detection, compatible method filtering, and simple visualization controls.
- Constraints from the guidelines: implement a unified interface with multiple models and XAI methods, provide comparison features, and deliver clear documentation of the system and its limitations.

## 3. System Overview and Architecture
The application follows a modular architecture that separates UI, model logic, and explainability logic:
- UI layer: `src/ui/pages.py` defines the Gradio interface with three tabs and binds user actions to backend functions.
- Core layer: `src/core` provides input detection, preprocessing utilities, and orchestration for prediction and explanation.
- Model layer: `src/models` implements wrappers for image and audio models with a common interface.
- XAI layer: `src/xai` implements XAI wrappers that output standardized explanation artifacts.

A registry system in `src/core/registry.py` centralizes available models and XAI methods. Each method declares compatible input types and, optionally, specific models. This registry drives dropdown choices and enforces compatibility.

Data flow is as follows:
- The user uploads a file, and the system detects input type from its extension.
- The registry provides compatible models and XAI methods for that input type.
- The selected model preprocesses the input and produces a prediction with a top-1 confidence.
- The selected XAI methods generate explanation artifacts.
- The UI renders overlays and heatmaps in the gallery, along with prediction summaries.

## 4. Supported Modalities and Models
### 4.1 Image Modality
- Models: AlexNet and DenseNet (TorchVision variants, two-class heads).
- Input format and preprocessing: .png/.jpg images are resized to 224x224, normalized with ImageNet mean and standard deviation, and converted to tensors.
- Output format: class label and top-1 softmax confidence. The labels are "normal" and "cancer" as defined in the image model wrapper.

### 4.2 Audio Modality
- Model: AudioCNN (a simple convolutional neural network).
- Input format: .wav audio files.
- Audio preprocessing pipeline: audio is loaded and normalized, optionally resampled to 16 kHz, converted to a mel spectrogram with 64 mel bins, normalized, and resized to 64x64 for the CNN. A waveform plot and a mel spectrogram image are prepared for display.
- Output format: class label and top-1 softmax confidence. The labels are "real" and "fake" as defined in the audio model wrapper.

## 5. Explainability Methods
### Grad-CAM
- Theoretical description: uses gradients of the target class to weight convolutional feature maps, producing a coarse heatmap of influential regions.
- Relevance: effective for convolutional image models where spatial localization is meaningful.
- Integration: implemented in `src/xai/gradcam_xai.py` using the last convolutional layer of the model wrapper.
- Supported modalities: image only.

### LIME
- Theoretical description: fits a local surrogate model around the input by perturbing features and measuring prediction changes.
- Relevance: model-agnostic and useful for explaining single predictions with interpretable approximations.
- Integration: implemented in `src/xai/lime_xai.py` using `lime_image` with a custom predict function. For audio, explanations are produced on mel spectrogram images.
- Supported modalities: image and audio (via mel spectrogram representation).

### SHAP
- Theoretical description: estimates feature contributions using Shapley values computed from perturbed samples.
- Relevance: provides principled attribution values that can be compared across features.
- Integration: implemented in `src/xai/shap_xai.py` with a kernel explainer over flattened inputs and a background of zeros. For audio, inputs are mel spectrogram images.
- Supported modalities: image and audio (via mel spectrogram representation).

### Integrated Gradients
- Theoretical description: integrates gradients along a straight path from a baseline to the input to estimate feature attributions.
- Relevance: gradient-based attribution that is less noisy than raw gradients for image models.
- Integration: implemented in `src/xai/extra_xai.py` with a zero baseline and 20 steps.
- Supported modalities: image only.

## 6. User Interface and Features
### 6.1 Single Explanation Tab
- Purpose: generate and inspect explanations for a single model and multiple selected XAI methods.
- Workflow: upload a file, select a model and methods, then run to view predictions and explanations.
- Controls: overlay toggle, opacity slider, view mode (original, heatmap, overlay), and zoom via center crop controls; audio view includes waveform and spectrogram with a time window slider.
- Output interpretation: each explanation is rendered as an image with a caption including model, method, prediction, and confidence.

### 6.2 Compare Tab
- Purpose: compare multiple XAI methods side by side for the same model and input.
- Display of explanations: a gallery shows explanation images for each selected method.
- Predictions alongside explanations: a prediction summary is shown, and each image includes a caption with the model, method, and top-1 prediction.
- Added value: supports qualitative comparison of how different methods highlight input regions.

### 6.3 Interactive Visualization Tab
- Purpose: provide an educational pointer to interactive exploration.
- Relationship with other tabs: it explains that interactive exploration is performed in the Single Explanation tab using overlay and zoom controls.
- Educational value: clarifies how to interpret and explore explanations rather than introducing new processing.

## 7. Compatibility Management
The registry and UI enforce compatibility by filtering models and XAI methods based on input type and optional model constraints. When a user uploads a file or changes the selected model, the XAI list is regenerated to include only valid methods. This prevents invalid combinations (for example, Grad-CAM on audio) and keeps the UI responsive and robust.

## 8. Results and Limitations
- Observations: the UI shows heatmaps and overlays that highlight regions or time-frequency areas that influence predictions, enabling qualitative inspection.
- Image vs audio explanations: image explanations use spatial heatmaps overlaid on the original image; audio explanations operate on mel spectrogram images rather than raw waveforms.
- Model weights: AlexNet, DenseNet, and AudioCNN are initialized with random weights in this repository. The outputs are intended for interface demonstration, not scientific or clinical interpretation. Replacing with trained weights is required for meaningful explanations.
- Known limitations: LIME and SHAP are approximate and can be noisy; SHAP uses small sample sizes for speed; Integrated Gradients and Grad-CAM are limited to image models; no quantitative evaluation is provided.

## 9. Compliance with Project Guidelines
The implementation aligns with the required features as follows:
- Unified interface for image and audio inputs: implemented via input-type detection and a single Gradio application.
- Multiple models per modality: AlexNet and DenseNet for images, AudioCNN for audio. Note that pretrained checkpoints are not included in this repository.
- Multiple XAI methods: Grad-CAM, LIME, SHAP, and Integrated Gradients are available through a common registry.
- Automatic filtering of incompatible methods: enforced by the registry and UI update logic.
- Comparison view: a dedicated Compare tab displays multiple explanations and prediction summaries.
- Documentation: README, compliance memo, and this report are provided.
- Optional or bonus elements implemented: interactive controls (overlay, opacity, zoom), audio waveform and spectrogram displays, and an Interactive Visualization guidance tab.

## 10. Conclusion and Perspectives
The project delivers a unified, modular XAI interface for both image and audio models, with multiple explanation techniques and comparison workflows. It provides educational value by allowing users to explore how different methods visualize model behavior, while keeping the codebase extensible through model and XAI registries. Future improvements could include:
- More models per modality and additional domains.
- Better-trained audio models and the inclusion of pretrained checkpoints.
- Additional XAI techniques such as occlusion sensitivity or attention-based explanations.
- Quantitative evaluation of explanation quality and user studies.


