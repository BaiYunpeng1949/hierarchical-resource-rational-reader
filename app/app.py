"""
Gradio front end for the hierarchical resource-rational reader.

Two tabs, matching the train / simulate split:

  Simulate -- pick a checkpoint for each of the three levels, set the time
              pressure and the cognitive free parameters, get scanpaths back.
  Train    -- run a small PPO job for one level, watch the reward curve, then
              use the result in Simulate or download it.

Run locally:   python app/app.py
"""
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import gradio as gr

import model_params
import model_registry as registry
import sim_runner
import trainer

TIME_CHOICES = ["30s", "60s", "90s"]

TUNABLE_PARAMS = [p for p in model_params.PARAMS if p.apply != "offline"]
PARAM_ORDER = [p.key for p in TUNABLE_PARAMS]

DEMO_MODEL_WARNING = (
    "Models trained here run for thousands of timesteps on CPU. The built-in "
    "checkpoints are 100-200M timesteps on GPU."
)


def _level_choices(level):
    choices = registry.choices_for_level(level)
    return choices


def _default_for(level):
    choices = registry.choices_for_level(level)
    for label, value in choices:
        if label.startswith("[built-in]"):
            return value
    return choices[0][1] if choices else None


def refresh_dropdowns():
    """Rebuild all three model dropdowns, preserving nothing but validity."""
    return tuple(
        gr.update(choices=_level_choices(level), value=_default_for(level))
        for level in registry.LEVELS
    )


# ----------------------------------------------------------------------------- simulate


def _asset_choices():
    """
    The n_parafov dropdown, with any missing variant marked rather than hidden.
    """
    choices = []
    for asset in model_params.OBSERVATION_ASSETS:
        label = asset.label if asset.exists else f"{asset.label}  [not generated]"
        choices.append((label, asset.key))
    return choices


def on_simulate(
    word_model, sentence_model, text_model,
    stimuli, conds, rho, w_skip, coverage, trials, heatmaps, asset_key,
    progress=gr.Progress(),
):
    model_ids = {
        "word_recognizer": word_model,
        "sentence_reader": sentence_model,
        "text_reader": text_model,
    }
    stimulus_ids = [int(stimuli)] if stimuli is not None and stimuli != "" else []

    def report(frac, msg):
        progress(frac, desc=msg)

    try:
        images, summary = sim_runner.run_simulation(
            model_ids=model_ids,
            stimulus_ids=stimulus_ids,
            time_conditions=[conds] if conds else [],
            rho=float(rho),
            w_skip=float(w_skip),
            coverage=float(coverage),
            trials=int(trials),
            make_heatmaps=bool(heatmaps),
            asset_key=asset_key,
            progress=report,
        )
    except Exception as exc:
        return [], f"**Simulation failed.**\n\n```\n{type(exc).__name__}: {exc}\n```"
    return images, summary


# ----------------------------------------------------------------------------- train


def on_train(
    level, timesteps, n_envs, learning_rate, gamma, run_name,
    # Ahead of *param_values on purpose: Gradio only detects a gr.Progress
    # default on positional parameters, and stops looking at the first *args.
    progress=gr.Progress(),
    *param_values,
):
    # The panel always submits every parameter, in PARAM_ORDER; the trainer drops
    # the ones that do not apply to the level being trained.
    model_parameters = dict(zip(PARAM_ORDER, param_values))

    def report(frac, steps, mean_reward):
        progress(frac, desc=f"{steps:,} steps | mean reward {mean_reward:.3f}")

    try:
        result = trainer.train_demo_model(
            level=level,
            total_timesteps=int(timesteps),
            n_envs=int(n_envs),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            run_name=run_name,
            model_parameters=model_parameters,
            report=report,
        )
    except Exception as exc:
        return (
            None,
            f"**Training failed.**\n\n```\n{type(exc).__name__}: {exc}\n```",
            None,
            *refresh_dropdowns(),
        )

    return (
        result["curve_png"],
        result["summary"],
        result["zip_path"],
        *refresh_dropdowns(),
    )


def on_upload(file_obj, level, name):
    path = getattr(file_obj, "name", file_obj)
    model_id, message = registry.register_upload(path, level, name)
    status = message if model_id else f"**Upload rejected.** {message}"
    return (status, *refresh_dropdowns())


# ----------------------------------------------------------------------------- ui


DEFAULT_TRAIN_LEVEL = "word_recognizer"


def _build_param_panel():
    """
    One group of model-parameter controls per level, only one visible at a time.
    """
    param_components = {}
    level_groups = {}

    for level in registry.LEVELS:
        with gr.Group(visible=(level == DEFAULT_TRAIN_LEVEL)) as group:
            level_params = model_params.params_for_level(level)
            if not level_params:
                gr.Markdown(
                    f"No Table 1 parameter reaches the {registry.LEVEL_LABELS[level].lower()} "
                    "environment during training. Its free parameter (w_RP / coverage) "
                    "is sampled per episode by design and set on the Simulate tab."
                )
            for param in level_params:
                if param.apply == "offline":
                    # Not shown here: it is a property of the precomputed asset,
                    # chosen on the Simulate tab.
                    continue
                info = param.describe()
                if param.note:
                    info = f"{info} {param.note}"
                param_components[param.key] = gr.Slider(
                    param.low, param.high,
                    value=param.default, step=param.step,
                    label=param.label, info=info,
                )
        level_groups[level] = group

    return param_components, level_groups


def build():
    with gr.Blocks(title="Hierarchical Resource-Rational Reader") as demo:
        gr.Markdown(
            "# Hierarchical Resource-Rational Reader\n"
            "Simulate human reading under time pressure with a three-level RL model: "
            "word recognition, sentence reading, and text reading."
        )

        with gr.Tab("Simulate"):
            gr.Markdown(
                "### Models\nPick a checkpoint for each level. "
                "Levels are independent, so a newly trained word recogniser can be "
                "combined with the built-in sentence and text readers."
            )
            with gr.Row():
                word_dd = gr.Dropdown(
                    label="Word recognition",
                    choices=_level_choices("word_recognizer"),
                    value=_default_for("word_recognizer"),
                )
                sent_dd = gr.Dropdown(
                    label="Sentence reading",
                    choices=_level_choices("sentence_reader"),
                    value=_default_for("sentence_reader"),
                )
                text_dd = gr.Dropdown(
                    label="Text reading",
                    choices=_level_choices("text_reader"),
                    value=_default_for("text_reader"),
                )
            refresh_btn = gr.Button("Refresh model list", size="sm")

            gr.Markdown("### Stimulus and time pressure")
            with gr.Row():
                stim_cb = gr.Dropdown(
                    label="Stimulus",
                    choices=[str(i) for i in range(sim_runner.NUM_STIMULI)],
                    value="0",
                    info="Bundled texts with precomputed appraisal scores.",
                )
                cond_cb = gr.Radio(
                    label="Time condition",
                    choices=TIME_CHOICES,
                    value="30s",
                    info="Total time the reader is given.",
                )

            gr.Markdown("### Cognitive free parameters")
            with gr.Row():
                rho_sl = gr.Slider(0.0, 1.0, value=0.29, step=0.01,
                                   label="rho - word vigor")
                wskip_sl = gr.Slider(0.0, 2.0, value=0.70, step=0.01,
                                     label="w_skip - skip tendency")
                cov_sl = gr.Slider(0.0, 3.0, value=1.30, step=0.01,
                                   label="coverage - coverage drive")
            with gr.Row():
                trials_sl = gr.Slider(1, 10, value=1, step=1, label="Trials per stimulus x condition")
                heat_cb = gr.Checkbox(label="Also render gaze heatmaps", value=False)

            gr.Markdown("### Parafoveal preview (n_parafov)")
            asset_dd = gr.Dropdown(
                label="Sentence observation asset",
                choices=_asset_choices(),
                value=model_params.DEFAULT_ASSET_KEY,
                info=(
                    "How many letters of the next word the predictability "
                    "estimates assume are visible. Baked into a precomputed "
                    "asset, so choosing a value swaps the file the sentence "
                    "reader observes."
                ),
            )

            sim_btn = gr.Button("Run simulation", variant="primary")
            sim_summary = gr.Markdown()
            sim_gallery = gr.Gallery(label="Scanpaths", columns=2, height=560)

            sim_btn.click(
                on_simulate,
                inputs=[word_dd, sent_dd, text_dd, stim_cb, cond_cb,
                        rho_sl, wskip_sl, cov_sl, trials_sl, heat_cb, asset_dd],
                outputs=[sim_gallery, sim_summary],
            )
            refresh_btn.click(refresh_dropdowns, outputs=[word_dd, sent_dd, text_dd])

        with gr.Tab("Train"):
            gr.Markdown(f"### Train a demo model\n{DEMO_MODEL_WARNING}")
            with gr.Row():
                level_dd = gr.Dropdown(
                    label="Level to train",
                    choices=[(registry.LEVEL_LABELS[l], l) for l in registry.LEVELS],
                    value=DEFAULT_TRAIN_LEVEL,
                )
                run_name_tb = gr.Textbox(
                    label="Run name (optional)",
                    placeholder="leave blank for an automatic name",
                )
            with gr.Row():
                steps_sl = gr.Slider(
                    1000, trainer.MAX_TIMESTEPS, value=trainer.DEFAULT_TIMESTEPS, step=1000,
                    label="Total timesteps",
                    info="Roughly 1000 steps per second on CPU.",
                )
                envs_sl = gr.Slider(1, 4, value=2, step=1, label="Parallel envs")
            with gr.Row():
                lr_num = gr.Number(value=1e-4, label="Learning rate")
                gamma_num = gr.Number(value=0.90, label="Gamma")

            gr.Markdown(
                "### Model parameters\n"
                "The cognitive constants of Extended Data Table 1, as opposed to "
                "the RL settings above: these change what the policy is trained "
                "*to do*, not how well it is fitted. Defaults are the values the "
                "model currently uses."
            )

            param_components, level_groups = _build_param_panel()

            gr.Markdown(
                "A trained checkpoint records the values it was trained under, "
                "and the Simulate tab applies them again when you pick it. "
                "**C_STM changes the observation space**, so a checkpoint trained "
                "at one C_STM cannot be loaded at another - keep those runs named "
                "distinctly."
            )

            train_btn = gr.Button("Train", variant="primary")
            train_summary = gr.Markdown()
            train_curve = gr.Image(label="Training curve", height=380)
            train_download = gr.File(label="Download this checkpoint (.zip)")

            gr.Markdown(
                "Trained checkpoints live in the model dropdowns on the Simulate tab "
                "for this session only. **They are lost when the app restarts** - "
                "download anything you want to keep."
            )

            gr.Markdown("### Upload a checkpoint")
            with gr.Row():
                up_file = gr.File(label="PPO checkpoint (.zip)", file_types=[".zip"])
                up_level = gr.Dropdown(
                    label="Which level is it for?",
                    choices=[(registry.LEVEL_LABELS[l], l) for l in registry.LEVELS],
                    value="word_recognizer",
                    info="Required - a checkpoint loaded at the wrong level fails.",
                )
                up_name = gr.Textbox(label="Name (optional)")
            up_btn = gr.Button("Upload")
            up_status = gr.Markdown()

            def show_params_for(level):
                return tuple(
                    gr.update(visible=(lvl == level)) for lvl in registry.LEVELS
                )

            level_dd.change(
                show_params_for,
                inputs=[level_dd],
                outputs=[level_groups[lvl] for lvl in registry.LEVELS],
            )

            train_btn.click(
                on_train,
                inputs=[level_dd, steps_sl, envs_sl, lr_num, gamma_num, run_name_tb]
                       + [param_components[key] for key in PARAM_ORDER],
                outputs=[train_curve, train_summary, train_download,
                         word_dd, sent_dd, text_dd],
            )
            up_btn.click(
                on_upload,
                inputs=[up_file, up_level, up_name],
                outputs=[up_status, word_dd, sent_dd, text_dd],
            )

    return demo


if __name__ == "__main__":
    build().queue().launch()
