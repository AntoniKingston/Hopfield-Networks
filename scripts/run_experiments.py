"""Run every Project 2 experiment and write tables/figures used by the report.

    uv run python scripts/run_experiments.py              # everything
    uv run python scripts/run_experiments.py --only q4 q6
    uv run python scripts/run_experiments.py --quick      # fewer trials, for a smoke test

Outputs: results/*.csv (raw data), report/tables/*.tex, report/figures/*.pdf.
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments as E
from hopfield import Hopfield
from hopfield_datasets import DATA_DIR, DATASETS, PROJECT_ROOT, load_dataset, load_images

RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "report" / "figures"
TABLES = PROJECT_ROOT / "report" / "tables"

RULE_PL = {
    "hebb": "Hebb",
    "oja": "Oja",
    "oja_subspace": "Oja podprz.",
    "delta": "delta",
    "projection": "projekcyjna",
}
# one-letter tags used above pattern thumbnails: H+ / O- / S+ / P+
RULE_ABBR = {"hebb": "H", "oja": "O", "oja_subspace": "S", "delta": "D", "projection": "P"}
RULE_COLORS = {
    "hebb": "tab:blue",
    "oja": "tab:orange",
    "oja_subspace": "tab:purple",
    "projection": "tab:green",
}
# the subspace rule converges to the projection rule, so its curve is dashed to stay visible
RULE_STYLE = {
    "hebb": {"marker": "o"},
    "oja": {"marker": "o"},
    "oja_subspace": {"marker": "s", "ls": "--"},
    "projection": {"marker": "o"},
}
# drawn last so that it is not hidden under the identical projection-rule curve
RULE_PLOT_ORDER = ["hebb", "oja", "projection", "oja_subspace"]
MODE_PL = {"sync": "synchr.", "async": "asynchr."}
OUTCOME_PL = {
    "exact": "oryginał",
    "stored": "wzorzec",
    "inverse": "negatyw",
    "other": "inny wzorzec",
    "spurious": "stan fałszywy",
    "cycle": "cykl",
    "unconverged": "brak zbieżności",
}
OUTCOME_COLORS = {
    "exact": "tab:green",
    "stored": "tab:green",
    "inverse": "tab:olive",
    "other": "tab:orange",
    "spurious": "tab:red",
    "cycle": "tab:purple",
    "unconverged": "tab:gray",
}

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9})


def save_table(df: pd.DataFrame, name: str, latex: pd.DataFrame | None = None, **to_latex):
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    table = df if latex is None else latex
    (TABLES / f"{name}.tex").write_text(
        table.to_latex(index=False, **{"escape": True, "float_format": "%.3f"} | to_latex),
        encoding="utf-8",
    )


def save_fig(fig, name: str):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def show(ax, pattern, shape, title=None, color=None):
    ax.imshow(np.asarray(pattern).reshape(shape), cmap="gray", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=7, color=color or "black")
    if color:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)


def pattern_grid(patterns, shape, titles=None, colors=None, cols=8, scale=1.2):
    n = len(patterns)
    cols = min(cols, n)
    rows = -(-n // cols)
    aspect = shape[0] / shape[1]
    fig, axes = plt.subplots(
        rows, cols, figsize=(scale * cols, scale * rows * aspect + 0.3 * rows), squeeze=False
    )
    for k, ax in enumerate(axes.flat):
        if k < n:
            show(
                ax, patterns[k], shape, titles[k] if titles else None, colors[k] if colors else None
            )
        else:
            ax.axis("off")
    return fig


# ---------- Q1/Q2 ----------


def datasets_table(args):
    data = E.dataset_table()
    latex = data[
        [
            "dataset",
            "shape",
            "P",
            "N",
            "load",
            "bias",
            "mean_abs_overlap",
            "max_abs_overlap",
            "cond",
        ]
    ].assign(cond=data["cond"].map(lambda v: f"{v:.0f}"))
    latex = latex.rename(
        columns={
            "dataset": "zbiór",
            "shape": "rozmiar",
            "load": "$P/N$",
            "bias": "śr. piksel",
            "mean_abs_overlap": "śr. $|m|$",
            "max_abs_overlap": "maks. $|m|$",
            "cond": "$\\kappa(XX^\\top)$",
        }
    )
    save_table(data, "datasets", latex, escape=False)
    return data


def q1(args):
    data = datasets_table(args)

    stab = E.stability_table()
    save_table(stab, "stability_long")
    pivot = stab.pivot(index="dataset", columns="rule", values="stable").reindex(DATASETS)
    strict = stab[stab.rule == "hebb"].set_index("dataset")["stable_strict"].reindex(DATASETS)
    table = pd.DataFrame(
        {
            "zbiór": DATASETS,
            "$P$": data.set_index("dataset").loc[DATASETS, "P"].values,
            "Hebb": pivot["hebb"].values,
            "Hebb (ścisła)": strict.values,
            "przesłuch": data.set_index("dataset").loc[DATASETS, "hebb_unstable_patterns"].values,
            "Oja": pivot["oja"].values,
            "Oja podprz.": pivot["oja_subspace"].values,
            "delta": pivot["delta"].values,
            "projekcyjna": pivot["projection"].values,
        }
    )
    table["przesłuch"] = table["$P$"] - table["przesłuch"]
    save_table(table, "stability", escape=False)

    for name in DATASETS:
        d = load_dataset(name)
        nets = {rule: E.trained_network(d, rule) for rule in E.RULES}
        masks = {rule: net.stable_patterns() for rule, net in nets.items()}
        titles = [
            " ".join(f"{RULE_ABBR[r]}{'+' if masks[r][k] else '-'}" for r in masks)
            for k in range(d.n_patterns)
        ]
        colors = ["tab:green" if masks["hebb"][k] else "tab:red" for k in range(d.n_patterns)]
        fig = pattern_grid(d.patterns, d.shape, titles, colors, cols=13 if d.n_patterns > 8 else 8)
        save_fig(fig, f"patterns_{name}")

    cap = E.random_capacity(trials=3 if args.quick else 20)
    save_table(cap, "random_capacity")
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for n, g in cap.groupby("N"):
        axes[0].plot(g.load, g.unstable_bits, "o-", label=f"N={n}")
        axes[1].plot(g.load, g.stable_patterns, "o-", label=f"N={n}")
        axes[2].plot(g.load, g.final_overlap, "o-", label=f"N={n}")
    from scipy.special import erfc

    loads = np.linspace(0.01, 0.3, 100)
    axes[0].plot(loads, 0.5 * erfc(np.sqrt(1 / (2 * loads))), "k--", label="teoria")
    for ax, title in zip(
        axes,
        ["udział niestabilnych bitów", "udział stabilnych wzorców", "nakładanie po zbieżności"],
        strict=True,
    ):
        ax.axvline(0.138, color="gray", ls=":")
        ax.set_xlabel("$P/N$")
        ax.set_title(title)
        ax.legend()
    save_fig(fig, "random_capacity")

    traj = E.oja_trajectory(epochs=20 if args.quick else 100)
    save_table(traj, "oja_trajectory")
    names = traj.dataset.unique()
    fig, axes = plt.subplots(2, len(names), figsize=(3.3 * len(names), 5), squeeze=False)
    for j, name in enumerate(names):
        sub = traj[(traj.dataset == name) & ~traj.diverged]
        for (init, c), g in sub.groupby(["init", "relative_rate"]):
            ls = "-" if init == "hebb" else "--"
            label = f"{'Hebb' if init == 'hebb' else 'losowa'}, η={c:g}/N"
            if traj[
                (traj.dataset == name) & (traj.init == init) & (traj.relative_rate == c)
            ].diverged.any():
                label += " (rozbieżne)"
            axes[0, j].plot(g.epoch, g.stable, ls, label=label)
            axes[1, j].plot(g.epoch, g.pc_alignment, ls, label=label)
        axes[0, j].set_title(name)
        axes[0, j].set_ylabel("stabilne wzorce")
        axes[1, j].set_ylabel(r"śr. $|\cos(w_i, v_1)|$")
        axes[1, j].set_xlabel("epoka")
    axes[0, -1].legend(fontsize=6, title="inicjalizacja", title_fontsize=6)
    save_fig(fig, "oja_trajectory")

    sub_traj = E.oja_subspace_trajectory(epochs=60 if args.quick else 300)
    sub_traj.to_csv(RESULTS / "oja_subspace_trajectory.csv", index=False)
    plot_oja_subspace_trajectory(sub_traj)

    plot_weight_spectrum()


def plot_weight_spectrum(name: str = "animals-14x9"):
    d = load_dataset(name)
    fig, ax = plt.subplots(figsize=(4, 2.6))
    for rule in RULE_PLOT_ORDER:
        s = np.linalg.svd(E.trained_network(d, rule).weights, compute_uv=False)
        ax.semilogy(
            s[:20] / s[0],
            ls=RULE_STYLE[rule].get("ls", "-"),
            marker=RULE_STYLE[rule]["marker"],
            ms=3,
            label=RULE_PL[rule],
            color=RULE_COLORS[rule],
        )
    ax.set_xlabel("numer wartości osobliwej")
    ax.set_ylabel("$\\sigma_k / \\sigma_1$")
    ax.set_title(f"Widmo macierzy wag ({name})")
    ax.legend()
    save_fig(fig, "weight_spectrum")


def plot_oja_subspace_trajectory(sub_traj: pd.DataFrame):
    names = sub_traj.dataset.unique()
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 5), squeeze=False)
    for j, name in enumerate(names):
        d = load_dataset(name)
        sub = sub_traj[(sub_traj.dataset == name) & ~sub_traj.diverged]
        for (init, c), g in sub.groupby(["init", "relative_rate"]):
            ls = "-" if init == "hebb" else "--"
            init_pl = "Hebb" if init == "hebb" else "losowa"
            label = f"{init_pl}, $\\eta={c:g}/\\lambda_{{\\max}}$"
            axes[0, j].plot(g.epoch, g.stable, ls, label=label)
            axes[1, j].semilogy(g.epoch, g.dist_projection, ls, label=label)
        axes[1, j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0, j].axhline(d.n_patterns, color="gray", ls=":", lw=0.8)
        axes[0, j].set_title(f"{name} (P={d.n_patterns})")
        axes[0, j].set_ylabel("stabilne wzorce")
        axes[1, j].set_ylabel(r"$\max|W - W_{\mathrm{proj}}|$")
        axes[1, j].set_xlabel("epoka")
    axes[0, -1].legend(fontsize=6, title="inicjalizacja", title_fontsize=6)
    save_fig(fig, "oja_subspace_trajectory")


# ---------- Q3 ----------


def q3(args):
    comp = E.compare_25x25_25x50()
    latex = comp[
        ["dataset", "N", "bias", "mean_abs_overlap", "max_abs_overlap", "hebb_unstable_bits"]
        + [f"stable_{r}" for r in E.RULES]
    ].rename(
        columns={
            "dataset": "zbiór",
            "bias": "śr. piksel",
            "mean_abs_overlap": "śr. $|m|$",
            "max_abs_overlap": "maks. $|m|$",
            "hebb_unstable_bits": "niestab. bity (Hebb)",
            "stable_hebb": "Hebb",
            "stable_oja": "Oja",
            "stable_oja_subspace": "Oja podprz.",
            "stable_projection": "proj.",
        }
    )
    save_table(comp, "compare_25x50", latex, escape=False)

    variants = E.variants_25x50()
    fig, axes = plt.subplots(2, len(variants), figsize=(3.2 * len(variants), 5.4))
    for j, d in enumerate(variants):
        show(axes[0, j], d.patterns[0], d.shape, d.name)
        M = d.patterns @ d.patterns.T / d.n_neurons
        im = axes[1, j].imshow(M, vmin=-1, vmax=1, cmap="coolwarm")
        for (a, b), v in np.ndenumerate(M):
            axes[1, j].text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6)
        axes[1, j].set_title("macierz nakładania $m_{\\mu\\nu}$")
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8)
    save_fig(fig, "compare_25x50")

    df = E.noise_recovery(variants, trials=5 if args.quick else 20)
    summary = E.summarize_recovery(df)
    save_table(summary, "compare_25x50_noise")
    plot_recovery(summary, [d.name for d in variants], "compare_25x50_noise")


# ---------- Q4/Q5 ----------


def plot_recovery(summary, names, fig_name, metric="exact", rules=RULE_PLOT_ORDER):
    fig, axes = plt.subplots(
        1, len(names), figsize=(2.6 * len(names), 2.6), sharey=True, squeeze=False
    )
    for j, name in enumerate(names):
        ax = axes[0, j]
        for rule in rules:
            for mode in E.MODES:
                g = summary[
                    (summary.dataset == name) & (summary.rule == rule) & (summary["mode"] == mode)
                ]
                ax.plot(
                    g.noise,
                    g[metric],
                    "-" if mode == "sync" else "--",
                    marker=RULE_STYLE[rule]["marker"],
                    ms=2.5,
                    color=RULE_COLORS[rule],
                    label=f"{RULE_PL[rule]}, {MODE_PL[mode]}",
                )
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("udział odwróconych pikseli")
        ax.set_ylim(-0.05, 1.05)
    axes[0, 0].set_ylabel({"exact": "odtworzono oryginał", "overlap": "nakładanie końcowe"}[metric])
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        fontsize=6,
    )
    save_fig(fig, fig_name)


def q4(args):
    datasets = [load_dataset(n) for n in DATASETS]
    df = E.noise_recovery(datasets, trials=3 if args.quick else 10)
    df.to_csv(RESULTS / "noise_recovery_raw.csv", index=False)
    summary = E.summarize_recovery(df)
    save_table(summary, "noise_recovery")

    for metric in ("exact", "overlap"):
        plot_recovery(summary, DATASETS[:4], f"noise_{metric}_a", metric)
        plot_recovery(summary, DATASETS[4:], f"noise_{metric}_b", metric)

    # critical noise level: largest p for which the projection rule still recovers >= 90 %
    def critical(g):
        ok = g[g.exact >= 0.9].noise
        return ok.max() if len(ok) else 0.0

    asyn = summary[summary["mode"] == "async"]
    crit = asyn.groupby(["dataset", "rule"]).apply(critical, include_groups=False).unstack()
    crit = crit.reindex(DATASETS)

    def at(rule, noise, metric):
        g = asyn[(asyn.rule == rule) & np.isclose(asyn.noise, noise)].set_index("dataset")
        return g.loc[DATASETS, metric].values

    stats = E.dataset_table().set_index("dataset")
    crit_table = pd.DataFrame(
        {
            "zbiór": DATASETS,
            "$P$": stats.loc[DATASETS, "P"].values,
            "śr. $|m|$": stats.loc[DATASETS, "mean_abs_overlap"].values,
            "H: dokł. 10\\%": at("hebb", 0.1, "exact"),
            "H: dokł. 30\\%": at("hebb", 0.3, "exact"),
            "H: nakł. 30\\%": at("hebb", 0.3, "overlap"),
            "Oja: nakł. 30\\%": at("oja", 0.3, "overlap"),
            "S: dokł. 30\\%": at("oja_subspace", 0.3, "exact"),
            "P: dokł. 30\\%": at("projection", 0.3, "exact"),
            "P: $p_{90}$": crit["projection"].values,
        }
    )
    save_table(crit_table, "critical_noise", escape=False, float_format="%.2f")

    # outcome breakdown for the Hebb rule, asynchronous mode
    hebb = df[(df.rule == "hebb") & (df["mode"] == "async")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharey=True)
    for ax, name in zip(axes.flat, DATASETS, strict=True):
        g = pd.crosstab(
            hebb[hebb.dataset == name].noise, hebb[hebb.dataset == name].outcome, normalize="index"
        )
        bottom = np.zeros(len(g))
        for outcome in ["exact", "inverse", "other", "spurious", "cycle", "unconverged"]:
            if outcome in g:
                ax.bar(
                    g.index,
                    g[outcome],
                    0.045,
                    bottom=bottom,
                    color=OUTCOME_COLORS[outcome],
                    label=OUTCOME_PL[outcome],
                )
                bottom += g[outcome].values
        ax.set_title(name, fontsize=8)
    axes[1, 0].set_xlabel("udział odwróconych pikseli")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, "noise_outcomes_hebb")

    # Q5: similar vs diverse sets of comparable size (5-7 patterns)
    # ordered from the most diverse to the most similar patterns
    q5_names = ["large-25x25", "animals-14x9", "OCRA-12x30-cut", "small-7x7", "large-25x50"]
    plot_recovery(summary, q5_names, "similar_vs_diverse", "exact")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, metric, label in [
        (axes[0], "exact", "odtworzono oryginał"),
        (axes[1], "overlap", "nakładanie końcowe"),
    ]:
        sub = summary[
            (summary["mode"] == "async") & (summary.rule == "hebb") & np.isclose(summary.noise, 0.1)
        ]
        sub = sub.set_index("dataset").join(stats)
        ax.scatter(sub.mean_abs_overlap, sub[metric], c=sub.P, cmap="viridis")
        for name, row in sub.iterrows():
            ax.annotate(f"{name} (P={row.P})", (row.mean_abs_overlap, row[metric]), fontsize=6)
        ax.set_xlabel("średnie $|m_{\\mu\\nu}|$ między wzorcami")
        ax.set_ylabel(f"{label} (Hebb, 10% szumu)")
    save_fig(fig, "overlap_vs_recovery")

    # example: recovery of one noisy pattern, both modes
    d = load_dataset("OCRA-12x30-cut")
    net = E.trained_network(d, "projection")
    rng = np.random.default_rng(E.SEED)
    x = net.corrupt(d.patterns[5], 0.4, rng)
    for mode in E.MODES:
        res = net.run(x, mode=mode, seed=E.SEED, frames_per_sweep=4)
        hist = res.history
        titles = [f"krok {k}\nE={net.energy(h):.1f}" for k, h in enumerate(hist)]
        cols = min(5, len(hist))  # the async run has 9 frames: rows of 5 + 4
        save_fig(pattern_grid(hist, d.shape, titles, cols=cols), f"convergence_{mode}")


# ---------- Q5 (controlled): grouped vs diverse sets with the same N and P ----------

Q5_SETS = ["zróżnicowany", "grupy q=0.2", "grupy q=0.1", "grupy q=0.05"]


def mean_std(g, col):
    return f"{g[col].mean():.1f} $\\pm$ {g[col].std(ddof=0):.1f}"


def q5(args):
    stab, summary = E.similar_vs_diverse_controlled(
        reps=2 if args.quick else 5, trials=2 if args.quick else 5
    )
    stab.to_csv(RESULTS / "similar_diverse_stability_raw.csv", index=False)
    save_table(summary, "similar_diverse_recovery")

    rows = []
    for name in Q5_SETS:
        g = stab[stab.set == name]
        row = {"zbiór": name, "$P$": g.P.iloc[0], "śr. $|m|$": f"{g.mean_abs_overlap.mean():.2f}"}
        for rule in E.RULES:
            row[RULE_PL[rule]] = mean_std(g[g.rule == rule], "stable")
        rows.append(row)
    table = pd.DataFrame(rows)
    save_table(table, "similar_diverse_stability", escape=False)
    plot_similar_diverse(stab, summary)
    group_prototypes_table(args)

    d = E.grouped_and_diverse_sets(rep=0)
    shown = [d[0].patterns, d[2].patterns]  # diverse and q = 0.1
    fig = pattern_grid(np.vstack(shown), d[0].shape, [d[0].name] * 9 + [d[2].name] * 9, cols=9)
    save_fig(fig, "similar_diverse_sets")


def group_prototypes_table(args):
    proto = E.group_prototype_attraction(reps=2 if args.quick else 5)
    latex = proto.assign(rule=proto.rule.map(RULE_PL)).rename(
        columns={
            "set": "zbiór",
            "rule": "reguła",
            "overlap_member": "nakł. z wariantem",
            "overlap_prototype": "nakł. z obrazem źródłowym",
        }
    )
    save_table(proto, "group_prototypes", latex, float_format="%.2f")


def plot_similar_diverse(stab: pd.DataFrame, summary: pd.DataFrame):
    fig, axes = plt.subplots(1, len(Q5_SETS), figsize=(2.8 * len(Q5_SETS), 2.7), sharey=True)
    for ax, name in zip(axes, Q5_SETS, strict=True):
        for rule in RULE_PLOT_ORDER:
            g = summary[(summary.set == name) & (summary.rule == rule)]
            ax.errorbar(
                g.noise,
                g.exact,
                yerr=g.exact_std,
                **RULE_STYLE.get(rule, {}),
                ms=3,
                capsize=2,
                color=RULE_COLORS[rule],
                label=RULE_PL[rule],
            )
        m = stab[stab.set == name].mean_abs_overlap.mean()
        ax.set_title(f"{name}\nśr. $|m|$ = {m:.2f}", fontsize=8)
        ax.set_xlabel("udział odwróconych pikseli")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("odtworzono oryginał (asynchr.)")
    axes[-1].legend(fontsize=6)
    save_fig(fig, "similar_diverse_controlled")


# ---------- repeatability ----------


def rep(args):
    tr = E.training_repeatability(seeds=range(3 if args.quick else 10))
    tr.to_csv(RESULTS / "training_repeatability_raw.csv", index=False)
    columns = [("hebb", "---"), ("projection", "---")] + [
        (r, i) for r in ("oja", "oja_subspace") for i in ("hebb", "random")
    ]
    rows = []
    for name in DATASETS:
        g = tr[tr.dataset == name].set_index(["rule", "init"])
        row = {"zbiór": name}
        for rule, init in columns:
            r = g.loc[(rule, init)]
            init_pl = {"hebb": "Hebb", "random": "losowa"}.get(init)
            label = RULE_PL[rule] if init == "---" else f"{RULE_PL[rule]} ({init_pl})"
            row[label] = (
                f"{r.stable_mean:.1f} $\\pm$ {r.stable_std:.1f}"
                if r.runs > 1
                else f"{r.stable_mean:.0f}"
            )
        rows.append(row)
    save_table(pd.DataFrame(rows), "training_repeatability", escape=False)

    rr = E.recall_repeatability(seeds=range(2 if args.quick else 5), trials=2 if args.quick else 5)
    save_table(rr, "recall_repeatability")
    plot_recall_repeatability(rr)


def plot_recall_repeatability(rr: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.8), sharey=True)
    for ax, name in zip(axes.flat, DATASETS, strict=True):
        for rule in RULE_PLOT_ORDER:
            g = rr[(rr.dataset == name) & (rr.rule == rule)]
            ax.errorbar(
                g.noise,
                g["mean"],
                yerr=g["std"],
                **RULE_STYLE.get(rule, {}),
                ms=3,
                capsize=2,
                color=RULE_COLORS[rule],
                label=RULE_PL[rule],
            )
        ax.set_title(name, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("udział odwróconych pikseli")
    for ax in axes[:, 0]:
        ax.set_ylabel("odtworzono oryginał")
    axes[0, -1].legend(fontsize=6)
    save_fig(fig, "recall_repeatability")


# ---------- Q6 ----------


def q6(args):
    df = E.random_input_outcomes(n_inputs=50 if args.quick else 200)
    summary = E.summarize_random(df)
    save_table(df.drop(columns="final"), "random_input_raw")
    save_table(summary, "random_input")

    order = ["stored", "inverse", "spurious", "cycle", "unconverged"]
    fig, axes = plt.subplots(1, len(E.RULES), figsize=(3.7 * len(E.RULES), 3.6), sharey=True)
    for ax, rule in zip(axes, E.RULES, strict=True):
        g = summary[(summary.rule == rule) & (summary["mode"] == "async")]
        g = g.set_index("dataset").reindex(DATASETS).reset_index()
        labels = list(g.dataset)
        bottom = np.zeros(len(g))
        for outcome in order:
            if outcome in g:
                vals = g[outcome].fillna(0).values
                ax.bar(
                    range(len(g)),
                    vals,
                    bottom=bottom,
                    color=OUTCOME_COLORS[outcome],
                    label=OUTCOME_PL[outcome],
                )
                bottom += vals
        ax.set_xticks(range(len(g)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"reguła {RULE_PL[rule]} (asynchr.)")
    axes[0].set_ylabel("udział losowych wejść")
    handles, names = axes[-1].get_legend_handles_labels()
    for ax in axes[:-1]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in names:
                handles.append(h)
                names.append(lab)
    fig.legend(handles, names, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    save_fig(fig, "random_input")

    latex = summary[summary["mode"] == "async"].copy()
    latex["rule"] = latex.rule.map(RULE_PL)
    cols = (
        ["dataset", "rule"]
        + [c for c in order if c in latex]
        + ["distinct_final_states", "mean_steps"]
    )
    latex = (
        latex[cols]
        .fillna(0)
        .rename(
            columns={
                "dataset": "zbiór",
                "rule": "reguła",
                "distinct_final_states": "różne stany",
                "mean_steps": "kroki",
            }
            | OUTCOME_PL
        )
    )
    save_table(summary, "random_input", latex, float_format="%.2f")

    # examples of where random inputs end up (Hebb, async)
    for name in ("animals-14x9", "large-25x25"):
        d = load_dataset(name)
        net = E.trained_network(d, "hebb")
        rng = np.random.default_rng(E.SEED)
        finals, titles = [], []
        for _ in range(8):
            res = net.run(net.random_state(rng), mode="async", seed=rng)
            m = net.overlaps(res.state)
            k = int(np.abs(m).argmax())
            finals.append(res.state)
            titles.append(f"{OUTCOME_PL[E.classify(net, res)]}\nm{k}={m[k]:+.2f}")
        save_fig(pattern_grid(finals, d.shape, titles, cols=8), f"random_examples_{name}")


# ---------- Q7 ----------


def q7(args):
    best, log = E.search_stable_5x5(
        iters=20_000 if args.quick else 100_000, attempts=3 if args.quick else 10
    )
    log.to_csv(RESULTS / "stable_5x5_search.csv", index=False)
    np.savetxt(RESULTS / "stable_5x5.csv", best, fmt="%d", delimiter=",")
    net = Hopfield((5, 5), best)
    net.hebbian_train()
    margins = net.stability_margins()
    assert net.stable_patterns(strict=True).all()
    titles = [f"#{k} min={margins[k].min() * 25:.0f}/25" for k in range(len(best))]
    save_fig(pattern_grid(best, (5, 5), titles, cols=9, scale=1.0), "stable_5x5")

    M = best @ best.T / 25
    off = np.abs(M[~np.eye(len(best), dtype=bool)])
    proj = E.projection_random_sets(trials=5 if args.quick else 50)
    save_table(proj, "projection_random_sets")
    summary = pd.DataFrame(
        [
            {
                "P": len(best),
                "mean_abs_overlap": off.mean(),
                "max_abs_overlap": off.max(),
                "min_margin_times_N": margins.min() * 25,
                "random_attraction_async": attraction_5x5(net),
            }
        ]
    )
    save_table(summary, "stable_5x5_summary")

    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.plot(proj.P, proj.hebb, "o-", label="Hebb (losowe wzorce)")
    ax.plot(proj.P, proj.projection, "s-", label="projekcyjna (losowe wzorce)")
    ax.axvline(len(best), color="tab:red", ls="--", label=f"znaleziony zbiór (Hebb), P={len(best)}")
    ax.set_xlabel("liczba wzorców P (N = 25)")
    ax.set_ylabel("P(wszystkie stabilne)")
    ax.legend(fontsize=6)
    save_fig(fig, "stable_5x5_random")


def attraction_5x5(net, noise=1, trials=200):
    """Fraction of 1-bit-corrupted inputs that return to the original (basin size check)."""
    rng = np.random.default_rng(E.SEED)
    ok = 0
    for _ in range(trials):
        mu = rng.integers(len(net.patterns))
        x = net.patterns[mu].copy()
        x[rng.choice(25, noise, replace=False)] *= -1
        ok += np.array_equal(net.run(x, mode="async", seed=rng).state, net.patterns[mu])
    return ok / trials


# ---------- Q8 ----------


def q8(args):
    o = E.oscillation_example()
    net, sync, asyn = o["net"], o["sync"], o["async"]
    # history is s, -s, s (cycle detected); one more real step shows the cycle continuing
    extra = net.recall(sync.history[-1])
    frames = [o["pattern"], *sync.history, extra]
    titles = ["wzorzec uczący"] + [
        f"synchr. krok {k}\nE={e:.2f}" for k, e in enumerate([*o["sync_energy"], net.energy(extra)])
    ]
    frames += asyn.history
    titles += [f"asynchr. krok {k}\nE={e:.2f}" for k, e in enumerate(o["async_energy"])]
    save_fig(pattern_grid(frames, net.shape, titles, cols=len(frames), scale=1.3), "oscillation")
    pd.DataFrame(
        {
            "mode": ["sync", "async"],
            "cycle_length": [sync.cycle_length, asyn.cycle_length],
            "converged": [sync.converged, asyn.converged],
            "steps": [sync.steps, asyn.steps],
            "final_matches_pattern": [net.match(sync.state), net.match(asyn.state)],
        }
    ).to_csv(RESULTS / "oscillation.csv", index=False)


# ---------- Q9 ----------


def q9(args):
    paths = sorted((DATA_DIR / "cats").iterdir())
    if not paths:
        raise SystemExit("no images in data/cats; run scripts/fetch_cats.py first")
    cats = load_images(paths, (200, 300), "median")
    names = [p.stem.split("_", 1)[-1] for p in paths]
    save_fig(pattern_grid(cats.patterns, cats.shape, names, cols=5, scale=2.2), "cats")

    t = time.time()
    res = E.large_bitmap_experiment(
        cats,
        noise_levels=(0.2, 0.4) if args.quick else (0.1, 0.2, 0.3, 0.4, 0.45),
        occlusions=(0.5,) if args.quick else (0.25, 0.5, 0.75),
    )
    print(f"  large-bitmap experiment: {time.time() - t:.0f}s")
    summary = E.summarize_recovery(res["noise"])
    save_table(summary, "cats_noise")
    occ = (
        res["occlusion"]
        .groupby(["rule", "occlusion"])
        .agg(exact=("outcome", lambda s: (s == "exact").mean()), overlap=("overlap", "mean"))
    )
    save_table(occ.reset_index(), "cats_occlusion")
    down = res["downsampled"].set_index("rule")
    stable = pd.DataFrame(
        [
            {
                "rozdzielczość": "300x200",
                "$N$": cats.n_neurons,
                "Hebb": res["stable"]["hebb"],
                "Oja": "---",
                "Oja podprz.": "---",
                "projekcyjna": res["stable"]["projection"],
                "Oja: śr. $|\\cos(w_i,v_1)|$": "---",
            },
            {
                "rozdzielczość": f"{res['small'].shape[1]}x{res['small'].shape[0]}",
                "$N$": res["small"].n_neurons,
                "Hebb": down.loc["hebb", "stable"],
                "Oja": down.loc["oja", "stable"],
                "Oja podprz.": down.loc["oja_subspace", "stable"],
                "projekcyjna": down.loc["projection", "stable"],
                "Oja: śr. $|\\cos(w_i,v_1)|$": f"{down.loc['oja', 'pc_alignment']:.2f}",
            },
        ]
    )
    save_table(stable, "cats_stable", escape=False)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for rule in ("hebb", "projection"):
        for mode in E.MODES:
            g = summary[(summary.rule == rule) & (summary["mode"] == mode)]
            axes[0].plot(
                g.noise,
                g.exact,
                "-" if mode == "sync" else "--",
                marker="o",
                label=f"{RULE_PL[rule]}, {MODE_PL[mode]}",
            )
        g = occ.loc[rule].reset_index()
        axes[1].plot(g.occlusion, g.exact, "o-", label=RULE_PL[rule])
    axes[0].set_xlabel("udział odwróconych pikseli")
    axes[1].set_xlabel("zasłonięta część obrazu (dół)")
    axes[0].set_ylabel("odtworzono oryginał")
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=6)
    save_fig(fig, "cats_recovery")

    nets = E.large_bitmap_networks(cats)
    rng = np.random.default_rng(E.SEED)
    frames, titles = [], []
    for corrupt_name, mu, x in [
        ("40% szumu", 0, nets[(cats.name, "hebb")].corrupt(cats.patterns[0], 0.4, rng)),
        ("zasłonięte 50%", 3, E.occlude(cats.patterns[3], cats.shape, 0.5, rng)),
    ]:
        frames.append(x)
        titles.append(corrupt_name)
        for rule in ("hebb", "projection"):
            out = nets[(cats.name, rule)].run(x, mode="async", seed=E.SEED, max_iter=20)
            outcome = OUTCOME_PL[E.classify(nets[(cats.name, rule)], out, mu)]
            frames.append(out.state)
            titles.append(f"{RULE_PL[rule]}: {outcome}, {out.steps} it.")
    save_fig(pattern_grid(frames, cats.shape, titles, cols=3, scale=2.4), "cats_examples")

    small = res["small"]
    masks = {r: E.trained_network(small, r).stable_patterns() for r in E.RULES}
    titles = [
        " ".join(f"{RULE_ABBR[r]}{'+' if masks[r][k] else '-'}" for r in E.RULES)
        for k in range(small.n_patterns)
    ]
    save_fig(pattern_grid(small.patterns, small.shape, titles, cols=5, scale=1.6), "cats_small")


SECTIONS = {
    "data": datasets_table,
    "q1": q1,
    "q3": q3,
    "q4": q4,
    "q5": q5,
    "q6": q6,
    "q7": q7,
    "q8": q8,
    "q9": q9,
    "rep": rep,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--quick", action="store_true", help="fewer trials (smoke test)")
    args = parser.parse_args()
    for d in (RESULTS, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
    print(f"seed = {E.SEED}")
    for name in args.only:
        t = time.time()
        print(f"[{name}] ...")
        SECTIONS[name](args)
        print(f"[{name}] done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
