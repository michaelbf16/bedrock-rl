"""Optional Matplotlib plots for committed training results."""

from __future__ import annotations

import math
from io import StringIO
from statistics import fmean


def _success_ticks(limit: float) -> list[float]:
    """Return readable percentage ticks while honoring the requested cap."""
    if limit <= 0.15:
        interval = 0.05
    elif limit <= 0.5:
        interval = 0.1
    else:
        interval = 0.25
    ticks = [index * interval
             for index in range(int(math.floor(limit / interval)) + 1)]
    if not math.isclose(ticks[-1], limit):
        ticks.append(limit)
    return ticks


def heldout_svg(summary: dict, success_max: float = 1.0) -> str:
    """Plot one paired held-out evaluation as a simple dark SVG."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    points = summary["points"]
    steps = [float(point["step"]) for point in points]
    success = [float(point["metrics"]["success_rate"])
               for point in points]
    background = "#171024"
    grid = "#332344"
    text = "#eee7f6"
    muted = "#a99ab8"
    amethyst = "#c79cff"
    with rc_context({
        "font.family": "DejaVu Sans Mono",
        "svg.fonttype": "none",
        "svg.hashsalt": "bedrock-rl-heldout",
        "axes.unicode_minus": False,
    }):
        figure = Figure(figsize=(10, 4.5), facecolor=background)
        FigureCanvasSVG(figure)
        axis = figure.subplots()
        axis.set_facecolor(background)
        axis.plot(
            steps,
            success,
            color=amethyst,
            linewidth=2.4,
            marker="o",
            markersize=6,
            markerfacecolor=amethyst,
            markeredgecolor=background,
            markeredgewidth=1.2,
            label="held-out eval",
            zorder=3,
        )
        span = max(steps) - min(steps)
        margin = max(1.0, span * 0.025)
        axis.set_xlim(min(steps) - margin, max(steps) + margin)
        axis.set_ylim(0, success_max)
        axis.set_xticks(steps)
        axis.set_yticks(_success_ticks(success_max))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis="y", color=grid, linewidth=0.8,
                  linestyle=(0, (1, 4)))
        axis.set_axisbelow(True)
        axis.tick_params(colors=muted, labelsize=9, length=0, pad=7)
        axis.set_xlabel("TRAINING STEP", color=muted, fontsize=9,
                        labelpad=10)
        axis.set_ylabel("HELD-OUT SUCCESS", color=muted, fontsize=9,
                        labelpad=10)
        for spine in axis.spines.values():
            spine.set_color(grid)
            spine.set_linewidth(1.1)
        axis.legend(
            loc="upper left",
            frameon=False,
            labelcolor=text,
            fontsize=8,
            handlelength=1.7,
        )
        figure.subplots_adjust(left=0.105, right=0.985,
                               bottom=0.17, top=0.96)
        output = StringIO()
        figure.savefig(
            output,
            format="svg",
            facecolor=background,
            metadata={"Creator": "bedrock-rl", "Date": None},
        )
        return "\n".join(
            line.rstrip() for line in output.getvalue().splitlines()) + "\n"


def combined_svg(training: dict, heldout, title: str,
                 success_max: float = 1.0) -> str:
    """Plot every training reward below held-out success checkpoints."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    heldout = list(heldout)
    points = training["points"]
    steps = [float(point["step"]) for point in points]
    scores = [float(point["score"]) for point in points]
    smooth = [
        fmean(scores[max(0, index - 4):index + 1])
        for index in range(len(scores))
    ]

    background = "#0e0a18"
    panel = "#120d20"
    grid = "#2d2040"
    text = "#f5effc"
    muted = "#9e90af"
    success_color = "#d8a7ff"
    reward_color = "#67e8f9"
    smooth_color = "#a78bfa"
    with rc_context({
        "font.family": "DejaVu Sans Mono",
        "svg.fonttype": "none",
        "svg.hashsalt": "bedrock-rl",
        "axes.unicode_minus": False,
    }):
        figure = Figure(figsize=(10, 5.8), facecolor=background)
        FigureCanvasSVG(figure)
        success_axis, reward_axis = figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": (3, 2),
                                            "hspace": 0.10})
        success_axis.set_facecolor(panel)
        reward_axis.set_facecolor(panel)

        reward_axis.plot(
            steps,
            scores,
            color=reward_color,
            linewidth=7,
            alpha=0.08,
            zorder=1,
        )
        reward_axis.plot(
            steps,
            scores,
            color=reward_color,
            linewidth=1.8,
            marker="o",
            markersize=4.2,
            markerfacecolor=reward_color,
            markeredgecolor=background,
            markeredgewidth=0.9,
            label="train reward (each step)",
            zorder=2,
        )
        reward_axis.plot(
            steps,
            smooth,
            color=smooth_color,
            linewidth=1.5,
            label="5-step mean",
            zorder=3,
        )

        colors = (success_color, "#7dd3fc", "#f0abfc")
        markers = ("o", "D", "^")
        for index, (label, summary) in enumerate(heldout):
            color = colors[index % len(colors)]
            eval_steps = [float(point["step"])
                          for point in summary["points"]]
            eval_scores = [float(point["metrics"]["success_rate"])
                           for point in summary["points"]]
            success_axis.plot(
                eval_steps,
                eval_scores,
                color=color,
                linewidth=8,
                alpha=0.09,
                zorder=4 + index,
            )
            success_axis.plot(
                eval_steps,
                eval_scores,
                color=color,
                linewidth=1.8,
                marker=markers[index % len(markers)],
                markersize=6.5,
                markerfacecolor=color,
                markeredgecolor=background,
                markeredgewidth=1.2,
                label=label,
                zorder=5 + index,
            )
            success_axis.fill_between(
                eval_steps, 0, eval_scores, color=color, alpha=0.035,
                zorder=1)
            for step, score in zip(eval_steps, eval_scores):
                success_axis.text(
                    step, score + success_max * 0.035,
                    f"{score * 100:.1f}%",
                    color=text, fontsize=6.8, ha="center", va="bottom",
                    zorder=8)

        all_steps = steps + [
            float(point["step"])
            for _label, summary in heldout
            for point in summary["points"]
        ]
        span = max(all_steps) - min(all_steps)
        margin = max(0.5, span * 0.025)
        success_axis.set_xlim(min(all_steps) - margin,
                              max(all_steps) + margin)
        success_axis.set_ylim(0, success_max)
        success_axis.set_yticks(_success_ticks(success_max))
        success_axis.yaxis.set_major_formatter(
            PercentFormatter(1, decimals=0))

        score_span = max(scores) - min(scores)
        score_pad = max(score_span * 0.15, 0.025)
        reward_low = math.floor((min(scores) - score_pad) * 20) / 20
        reward_high = math.ceil((max(scores) + score_pad) * 20) / 20
        reward_axis.set_ylim(reward_low, reward_high)
        reward_axis.set_yticks([
            reward_low + index * 0.05
            for index in range(round((reward_high - reward_low) / 0.05) + 1)
        ])
        reward_axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:+.2f}"))
        reward_axis.fill_between(
            steps, reward_low, scores, color=reward_color, alpha=0.035,
            zorder=0)
        for step, score in zip(steps, scores):
            reward_axis.text(
                step, score + 0.008, f"{score:.2f}",
                color=text, fontsize=6.2, ha="center", va="bottom",
                zorder=5)
        last_step = int(max(all_steps))
        tick_stride = 1 if last_step <= 20 else 10
        reward_axis.set_xticks(range(0, last_step + 1, tick_stride))

        for axis in (success_axis, reward_axis):
            axis.grid(axis="y", color=grid, linewidth=0.75,
                      linestyle=(0, (1, 5)), alpha=0.9)
            axis.set_axisbelow(True)
            axis.tick_params(colors=muted, labelsize=8, length=0, pad=7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(grid)
            axis.spines["left"].set_linewidth(1)
            axis.spines["bottom"].set_color(grid)
            axis.spines["bottom"].set_linewidth(1)
        success_axis.tick_params(axis="x", labelbottom=False)
        reward_axis.set_xlabel("TRAINING STEP", color=muted, fontsize=8,
                               labelpad=10)
        success_axis.set_ylabel("HELD-OUT SUCCESS", color=muted,
                                fontsize=8, labelpad=10)
        reward_axis.set_ylabel("TRAIN MEAN REWARD", color=muted,
                               fontsize=8, labelpad=10)
        success_axis.set_title(
            f"BEDROCK-RL  //  {title.upper()}",
            loc="left",
            color=text,
            fontsize=10,
            fontweight="bold",
            pad=14,
        )
        success_axis.legend(
            loc="lower right",
            frameon=False,
            labelcolor=text,
            fontsize=7.5,
            handlelength=1.8,
        )
        reward_axis.legend(
            loc="upper left",
            ncol=2,
            frameon=False,
            labelcolor=text,
            fontsize=7.5,
            handlelength=1.8,
            columnspacing=1.5,
        )
        figure.subplots_adjust(left=0.11, right=0.98,
                               bottom=0.11, top=0.90)

        output = StringIO()
        figure.savefig(
            output,
            format="svg",
            facecolor=background,
            bbox_inches="tight",
            pad_inches=0.12,
            metadata={"Creator": "bedrock-rl", "Date": None},
        )
        return "\n".join(
            line.rstrip() for line in output.getvalue().splitlines()) + "\n"
