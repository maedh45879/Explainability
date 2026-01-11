from __future__ import annotations

from typing import List, Tuple

import numpy as np
import gradio as gr
from PIL import Image

from ..core.io import detect_input_type
from ..core.inference import run_prediction, run_explanations
from ..core.registry import get_models_for_input_type, get_xai_for, list_models
from ..utils.errors import UserFacingError, friendly_error
from ..utils.viz import overlay_heatmap, crop_center


def build_app():
    with gr.Blocks(title="Unified XAI Lab") as demo:
        gr.Markdown("# Unified XAI Lab")
        gr.Markdown("Upload an audio (.wav) or image (.png/.jpg) and compare explainability methods.")
        gr.Markdown("Audio pipeline inspired by Deepfake-Audio-Detection-with-XAI (mel-spectrogram + real/fake).")
        gr.Markdown("Image pipeline inspired by LungCancerDetection (AlexNet/DenseNet + Grad-CAM).")

        with gr.Tabs():
            with gr.TabItem("Single Explanation"):
                _single_tab()
            with gr.TabItem("Compare"):
                _compare_tab()
            with gr.TabItem("Interactive Viz"):
                _interactive_tab()

    return demo


def _single_tab():
    with gr.Row():
        file_input = gr.File(label="Upload audio or image")
        input_type = gr.Textbox(label="Detected type", interactive=False)

    model_dropdown = gr.Dropdown(label="Model", choices=[], value=None)
    xai_dropdown = gr.Radio(label="XAI method", choices=[])
    gr.Markdown(_xai_help_text())
    compat_msg = gr.Markdown("")
    run_btn = gr.Button("Run", interactive=False)

    with gr.Row():
        pred_label = gr.Textbox(label="Prediction", interactive=False)
        pred_conf = gr.Textbox(label="Confidence", interactive=False)

    with gr.Row():
        overlay_toggle = gr.Checkbox(label="Overlay", value=True)
        opacity_slider = gr.Slider(label="Opacity", minimum=0.0, maximum=1.0, value=0.5)
        view_mode = gr.Radio(label="View", choices=["overlay", "heatmap", "original"], value="overlay")

    with gr.Row():
        crop_size = gr.Slider(label="Zoom (crop size)", minimum=0.2, maximum=1.0, value=1.0)
        crop_x = gr.Slider(label="Center X", minimum=0.0, maximum=1.0, value=0.5)
        crop_y = gr.Slider(label="Center Y", minimum=0.0, maximum=1.0, value=0.5)

    explanations_gallery = gr.Gallery(label="Explanations", columns=2, rows=2)
    audio_waveform = gr.Image(label="Waveform", visible=False)
    audio_spectrogram = gr.Image(label="Spectrogram", visible=False)
    audio_range = gr.Slider(label="Audio time window (seconds)", minimum=0.0, maximum=5.0, value=5.0, step=0.1, visible=False)

    file_input.change(
        fn=_on_file_change,
        inputs=[file_input],
        outputs=[input_type, model_dropdown, xai_dropdown, audio_waveform, audio_spectrogram, audio_range, run_btn, compat_msg],
    )
    model_dropdown.change(
        fn=_on_model_change_single,
        inputs=[input_type, model_dropdown],
        outputs=[xai_dropdown, run_btn, compat_msg],
    )
    run_btn.click(
        fn=_run_single,
        inputs=[file_input, input_type, model_dropdown, xai_dropdown, overlay_toggle, opacity_slider, view_mode, crop_size, crop_x, crop_y, audio_range],
        outputs=[pred_label, pred_conf, explanations_gallery, audio_waveform, audio_spectrogram],
    )


def _compare_tab():
    with gr.Row():
        file_input = gr.File(label="Upload audio or image")
        input_type = gr.Textbox(label="Detected type", interactive=False)

    model_dropdown = gr.Dropdown(label="Model", choices=[], value=None)
    xai_dropdown = gr.CheckboxGroup(label="XAI methods", choices=[])
    gr.Markdown(_xai_help_text())
    gr.Markdown(_compare_micro_help())
    compat_msg = gr.Markdown("")
    run_btn = gr.Button("Compare", interactive=False)

    pred_summary = gr.Markdown("### Prediction summary\nUpload a file and run compare to see predictions.")
    compare_gallery = gr.Gallery(label="Side-by-side", columns=2, rows=2)

    file_input.change(
        fn=_on_file_change_compare,
        inputs=[file_input],
        outputs=[input_type, model_dropdown, xai_dropdown, run_btn, compat_msg],
    )
    model_dropdown.change(
        fn=_on_model_change_multi,
        inputs=[input_type, model_dropdown],
        outputs=[xai_dropdown, run_btn, compat_msg],
    )
    run_btn.click(
        fn=_run_compare,
        inputs=[file_input, input_type, model_dropdown, xai_dropdown],
        outputs=[pred_summary, compare_gallery],
    )


def _interactive_tab():
    gr.Markdown("### Interactive Explanation Viewer")
    gr.Markdown("This demo lets you overlay and zoom any image explanation in the other tabs.")
    gr.Markdown("Use the Single Explanation tab to generate an overlay, then use its controls to explore it.")


def _on_file_change(file_obj):
    if file_obj is None:
        return (
            "",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.Image(visible=False),
            gr.Image(visible=False),
            gr.Slider(visible=False),
            gr.update(interactive=False),
            "",
        )
    try:
        input_type = detect_input_type(file_obj.name)
        models = get_models_for_input_type(input_type)
        model_choices = [m.name for m in models]
        xai_choices = [x.name for x in get_xai_for(input_type, model_choices[0])] if model_choices else []
        audio_visible = input_type == "audio"
        model_update = gr.update(choices=model_choices, value=model_choices[0]) if model_choices else gr.update(choices=[], value=None)
        xai_update = gr.update(choices=xai_choices, value=xai_choices[0]) if xai_choices else gr.update(choices=[], value=None)
        compat_msg = _compat_message_for_choices(input_type, model_choices, xai_choices)
        run_update = gr.update(interactive=bool(model_choices and xai_choices))
        return (
            input_type,
            model_update,
            xai_update,
            gr.Image(visible=audio_visible),
            gr.Image(visible=audio_visible),
            gr.Slider(visible=audio_visible, maximum=5.0),
            run_update,
            compat_msg,
        )
    except UserFacingError as exc:
        return (
            friendly_error(str(exc)),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.Image(visible=False),
            gr.Image(visible=False),
            gr.Slider(visible=False),
            gr.update(interactive=False),
            "",
        )


def _on_file_change_compare(file_obj):
    if file_obj is None:
        return "", gr.update(choices=[], value=None), gr.update(choices=[], value=[]), gr.update(interactive=False), ""
    try:
        input_type = detect_input_type(file_obj.name)
        models = get_models_for_input_type(input_type)
        model_choices = [m.name for m in models]
        xai_choices = [x.name for x in get_xai_for(input_type, model_choices[0])] if model_choices else []
        model_update = gr.update(choices=model_choices, value=model_choices[0]) if model_choices else gr.update(choices=[], value=None)
        xai_update = gr.update(choices=xai_choices, value=xai_choices[:1]) if xai_choices else gr.update(choices=[], value=[])
        compat_msg = _compat_message_for_choices(input_type, model_choices, xai_choices)
        run_update = gr.update(interactive=bool(model_choices and xai_choices))
        return (
            input_type,
            model_update,
            xai_update,
            run_update,
            compat_msg,
        )
    except UserFacingError as exc:
        return friendly_error(str(exc)), gr.update(choices=[], value=None), gr.update(choices=[], value=[]), gr.update(interactive=False), ""


def _on_model_change_single(input_type: str, model_name: str):
    if not input_type or not model_name:
        return gr.update(choices=[], value=None), gr.update(interactive=False), _compat_message_for_model(input_type, model_name, [])
    xai_choices = [x.name for x in get_xai_for(input_type, model_name)]
    if not xai_choices:
        return (
            gr.update(choices=[], value=None),
            gr.update(interactive=False),
            _compat_message_for_model(input_type, model_name, []),
        )
    return gr.update(choices=xai_choices, value=xai_choices[0]), gr.update(interactive=True), ""


def _on_model_change_multi(input_type: str, model_name: str):
    if not input_type or not model_name:
        return gr.update(choices=[], value=[]), gr.update(interactive=False), _compat_message_for_model(input_type, model_name, [])
    xai_choices = [x.name for x in get_xai_for(input_type, model_name)]
    if not xai_choices:
        return (
            gr.update(choices=[], value=[]),
            gr.update(interactive=False),
            _compat_message_for_model(input_type, model_name, []),
        )
    return gr.update(choices=xai_choices, value=xai_choices[:1]), gr.update(interactive=True), ""


def _run_single(file_obj, input_type: str, model_name: str, xai_method: str, overlay_on: bool,
                opacity: float, view: str, crop_size: float, crop_x: float, crop_y: float, audio_window: float):
    if file_obj is None:
        return "", "", [], None, None
    if not model_name:
        return friendly_error("Select a model."), "", [], None, None
    models = {m.name: m for m in list_models()}
    model = models[model_name]
    sample = model.preprocess(file_obj.name)
    pred = run_prediction(model, sample)
    xai_wrappers = [x for x in get_xai_for(input_type, model_name) if x.name == xai_method] if xai_method else []
    if not xai_wrappers:
        return pred.label, f"{pred.top1:.3f}", [], None, None
    explanations = run_explanations(model, sample, pred, xai_wrappers)

    rendered = []
    for expl in explanations:
        img = _select_view_image(expl.renderable, overlay_on, opacity, view)
        if img is not None:
            img = crop_center(img, crop_x, crop_y, crop_size)
            rendered.append((img, _format_caption(model_name, expl.method_name, pred)))

    wave_img, spec_img = _audio_visuals(sample, audio_window)
    return pred.label, f"{pred.top1:.3f}", rendered, wave_img, spec_img


def _run_compare(file_obj, input_type: str, model_name: str, xai_methods: List[str]):
    if file_obj is None or not model_name:
        return "### Prediction summary\nSelect a file and model to compare.", []
    models = {m.name: m for m in list_models()}
    model = models[model_name]
    sample = model.preprocess(file_obj.name)
    pred = run_prediction(model, sample)
    xai_wrappers = [x for x in get_xai_for(input_type, model_name) if x.name in (xai_methods or [])]
    explanations = run_explanations(model, sample, pred, xai_wrappers)
    rendered = []
    for expl in explanations:
        img = expl.renderable.get("overlay") or expl.renderable.get("heatmap") or expl.renderable.get("original")
        if img is not None:
            rendered.append((img, _format_caption(model_name, expl.method_name, pred)))
    summary = _format_prediction_summary(pred, model_name, [x.name for x in xai_wrappers])
    return summary, rendered


def _select_view_image(renderable, overlay_on: bool, opacity: float, view: str):
    if not renderable:
        return None
    base = renderable.get("original")
    heat = renderable.get("heatmap")
    overlay = renderable.get("overlay")
    if view == "original":
        return base
    if view == "heatmap":
        return heat
    if overlay_on and base is not None and heat is not None:
        return overlay_heatmap(base, np.array(heat.convert("L")), alpha=opacity)
    return overlay or base


def _audio_visuals(sample, audio_window: float):
    if sample.input_type != "audio":
        return None, None
    waveform = sample.metadata.get("waveform")
    sr = sample.metadata.get("sr")
    mel_img = sample.metadata.get("mel_image")
    if waveform is None or sr is None:
        return None, None
    total_secs = waveform.shape[-1] / float(sr)
    window = min(audio_window, total_secs)
    end = int(window * sr)
    segment = waveform[0, :end]
    wave_img = _plot_waveform(segment, sr)
    spec_img = Image.fromarray(mel_img).convert("RGB") if isinstance(mel_img, np.ndarray) else mel_img
    return wave_img, spec_img


def _plot_waveform(waveform: np.ndarray, sr: int) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2))
    times = np.arange(len(waveform)) / sr
    ax.plot(times, waveform, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[:, :, :3]
    plt.close(fig)
    return Image.fromarray(img)


def _format_prediction_summary(pred, model_name: str, xai_methods: List[str]) -> str:
    if not pred:
        return "### Prediction summary\nNo prediction available."
    methods = xai_methods or ["(none selected)"]
    rows = "\n".join(
        [f"| {model_name} | {method} | {pred.label} | {pred.top1:.3f} |" for method in methods]
    )
    return (
        "### Prediction summary\n"
        "| Model | Method | Pred | Confidence |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n"
        "\n_Confidence is the model softmax probability for the top-1 class._"
    )


def _format_caption(model_name: str, method_name: str, pred) -> str:
    if not pred:
        return f"Model: {model_name} | Method: {method_name}"
    return f"Model: {model_name} | Method: {method_name} | Pred: {pred.label} ({pred.top1:.3f})"


def _xai_help_text() -> str:
    return (
        "Method guide:\n"
        "- Grad-CAM: highlights spatial regions most responsible for the prediction.\n"
        "- LIME: fits a local surrogate model to explain the prediction for this input.\n"
        "- SHAP: estimates feature contributions using Shapley values over perturbed samples.\n"
        "- Integrated Gradients: attributes importance by integrating gradients from a baseline."
    )


def _compare_micro_help() -> str:
    return (
        "Micro-guide: Grad-CAM highlights regions influential for the predicted class. "
        "LIME shows perturbation-based importance. SHAP approximates feature contributions."
    )


def _compat_message_for_choices(input_type: str, model_choices: List[str], xai_choices: List[str]) -> str:
    if not input_type:
        return ""
    if not model_choices:
        return "No models available for this input type."
    if not xai_choices:
        return "No compatible XAI methods for this model/input. Select another model."
    return ""


def _compat_message_for_model(input_type: str, model_name: str, xai_choices: List[str]) -> str:
    if not input_type:
        return "Upload a file to load models and methods."
    if not model_name:
        return "Select a model to load compatible XAI methods."
    if not xai_choices:
        return "No compatible XAI methods for this model/input. Select another model."
    return ""
