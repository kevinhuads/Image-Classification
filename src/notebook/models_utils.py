import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

sns.set_theme(
    style="darkgrid",           
    rc={
        "figure.facecolor": "#0d1b2a",
        "axes.facecolor":   "#0d1b2a",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#2a3f5f",
        "axes.labelcolor":  "#ffffff",
        "text.color":       "#ffffff",
        "xtick.color":      "#ffffff",
        "ytick.color":      "#ffffff",
    },
    palette="deep"               
)

# ----------------------------
# Small color helpers
# ----------------------------

def lighten_color(rgb, amount=0.2):
    """Return a lighter RGB tuple by mixing with white.
       amount in [0,1], higher -> lighter."""
    r, g, b = rgb
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return (r, g, b)

def darken_color(rgb, amount=0.15):
    """Return a darker RGB tuple by scaling towards black.
       amount in [0,1], higher -> darker."""
    r, g, b = rgb
    r = r * (1.0 - amount)
    g = g * (1.0 - amount)
    b = b * (1.0 - amount)
    return (r, g, b)


def _fmt_arch_label(name: str) -> str:
    m = re.search(r"^(.*)_b(8|16|32)$", name)
    if m:
        base, bs = m.groups()
        return f"{base}\n(bs={bs})"
    return name


def get_list_and_df(dict_):
    scalars = {}
    lists = {}
    for k, v in dict_.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            lists[k.removeprefix("per_class_")] = v
        else:
            scalars[k] = float(v) if v is not None else np.nan
    df = pd.DataFrame(lists)
    return scalars, df


def _load_metrics_csv(path):
    """
    Robust CSV reader for training logs/metrics.
    - Auto-detects delimiter (comma/semicolon/tab) with the Python engine.
    - Ignores comment lines starting with '#'.
    - Skips malformed rows with irregular field counts.
    - Normalizes percentage-like columns (e.g., '85.2%') to floats in [0, 1] for *_acc*.
    """
    df = pd.read_csv(
        path,
        sep=None,               # sniff delimiter
        engine="python",        # required for sep=None
        comment="#",            # ignore commented log lines
        on_bad_lines="skip",    # drop rows with inconsistent columns
        encoding="utf-8",
        encoding_errors="ignore",
    )

    # Standardize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Ensure epoch is numeric
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    # Normalize accuracy columns: accept raw floats or strings like '85.2%' or '0.852'
    for col in ("train_acc1","val_acc1","train_acc5","val_acc5"):
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            has_pct = s.str.endswith("%")
            s = s.str.rstrip("%")
            s = pd.to_numeric(s, errors="coerce")
            # If there were percent signs, interpret as percent; else assume already in [0,1] or [0,100]
            if has_pct.any():
                s = s / 100.0
            # If max>1 and <=100, assume percent points; convert to [0,1]
            elif s.max(skipna=True) and 1.0 < float(s.max(skipna=True)) <= 100.0:
                s = s / 100.0
            df[col] = s

    # Drop rows with missing essentials
    return df.dropna(subset=["epoch"], how="any")

def topk_per_class_accuracies(y_true, y_score, ks = range(1,6)):
    """
    y_true: (n,) true labels (ints 0..C-1)
    y_score: (n, C) score matrix (higher = more likely)
    ks: int or iterable of ints
    returns: DataFrame indexed by class [0..C-1], columns 'topk_accuracy' with per-class top-k accuracy
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(y_score)
    if scores.ndim != 2:
        raise ValueError("y_score must be 2-D (n_samples, n_classes)")
    ks = sorted({int(k) for k in (ks if hasattr(ks, "__iter__") else [ks])})
    n_classes = scores.shape[1]
    ranked = np.argsort(scores, axis=1)[:, ::-1]          # descending class ranks per sample
    ks = [min(k, n_classes) for k in ks]
    classes = np.arange(n_classes)
    data = {}
    for k in ks:
        topk = ranked[:, :k]
        vals = []
        for c in classes:
            mask = (y_true == c)
            if not mask.any():
                vals.append(np.nan)
            else:
                hits = np.count_nonzero(np.any(topk[mask] == c, axis=1))
                vals.append(hits / mask.sum())
        data[f"top{k}_acc"] = vals
    return pd.DataFrame(data, index=classes)

def plot_tv(ax, df, x, y_train = None, y_val = None, *, scale=1, ms=6, ylabel=None,title=None):
    if y_train is not None:
        ax.plot(df[x], df[y_train] * scale, marker='o', markersize=ms, label=y_train)
    if y_val is not None:
        ax.plot(df[x], df[y_val]   * scale, marker='o', markersize=ms, label=y_val)
    ax.set_xlabel(x)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    
def plot_summary(path, epoch_dir, big_title="Training/Validation Accuracy Summary", ms=6):
    df = pd.read_csv(os.path.join(epoch_dir, path))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    plot_tv(axes[0], df, 'epoch', 'train_acc1', 'val_acc1', ms=ms,
            scale=100, ylabel='Top-1 Accuracy (%)',
            title='Top-1 Accuracy (Train vs Val)')

    plot_tv(axes[1], df, 'epoch', 'train_acc5', 'val_acc5', ms=ms,
            scale=100, ylabel='Top-5 Accuracy (%)',
            title='Top-5 Accuracy (Train vs Val)')

    # Big title for the whole figure
    fig.suptitle(big_title, fontsize=20, fontweight='bold')

    plt.show()
    
def plot_arches(epoch_dir, arch2cat, suffix, title, min_epoch=0, max_epoch=None,
                families=None, switch_legend=False, hide_family_legend=False):

    # normalize families param
    if families is not None:
        if isinstance(families, str):
            families = [families]
        families = set(families)

    # canonical architecture order from arch2cat keys (stable across runs)
    canonical_arches = list(arch2cat.keys())

    # determine which architectures to keep using canonical_arches order
    if families is None:
        arch_list = canonical_arches[:]
    else:
        def keep(a):
            fam = arch2cat.get(a, None)

            return (fam in families)
        arch_list = [a for a in canonical_arches if keep(a)]

    if len(arch_list) == 0:
        raise ValueError("No architectures selected after applying the families filter.")

    # stable category list and color mapping based on the full arch2cat mapping
    categories_full = sorted({v for v in arch2cat.values() if v is not None})
    cat_colors = {cat: f"C{i}" for i, cat in enumerate(categories_full)}

    # stable per-architecture marker and linestyle assignment using canonical_arches
    from itertools import cycle
    marker_cycle = cycle(['o', 's', '^', 'D', 'v', 'P', 'X', '*'])
    ls_cycle = cycle(['-', '--', '-.', ':'])
    arch_marker = {}
    arch_ls = {}
    for a in canonical_arches:
        arch_marker[a] = next(marker_cycle)
        arch_ls[a] = next(ls_cycle)

    # collect data
    df_pt = pd.DataFrame()
    for arch in canonical_arches:  # read in canonical order so loading is stable
        if arch not in arch_list:
            continue
        path = os.path.join(epoch_dir, f"{arch}_{suffix}.csv")
        try:
            df = _load_metrics_csv(path)
        except Exception:
            # skip architectures with missing or unreadable CSVs but continue with others
            continue
        df["arch"] = arch
        df_pt = pd.concat([df_pt, df[["arch", "epoch", "val_acc1", "val_acc5"]]], ignore_index=True)

    if df_pt.empty:
        raise ValueError("No metric data loaded for the selected architectures. Check files and suffix.")

    if max_epoch is None:
        max_epoch = int(df_pt["epoch"].max())
    df_pt = df_pt[(df_pt["epoch"] >= min_epoch) & (df_pt["epoch"] <= max_epoch)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

    # store plotted line handles in canonical order for a stable legend
    plotted_handles = []
    plotted_labels = []

    # iterate canonical_arches to keep plotting order stable and consistent across filtered/unfiltered runs
    for arch in canonical_arches:
        if arch not in arch_list:
            continue
        g = df_pt[df_pt["arch"] == arch].sort_values("epoch")
        if g.empty:
            continue
        fam = arch2cat.get(arch, None)
        color = cat_colors.get(fam, "C0")
        label = _fmt_arch_label(arch)
        line1, = axes[0].plot(g['epoch'], g['val_acc1'] * 100, marker=arch_marker[arch],
                              linestyle=arch_ls[arch], color=color, label=label)
        line2, = axes[1].plot(g['epoch'], g['val_acc5'] * 100, marker=arch_marker[arch],
                              linestyle=arch_ls[arch], color=color, label=label)
        # only add one handle/label per architecture (Top-1 handle used for legend)
        plotted_handles.append(line1)
        plotted_labels.append(label)

    axes[0].set_title('Validation Top-1 Accuracy')
    axes[1].set_title('Validation Top-5 Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].set_ylabel('Val Acc1 (%)')
    axes[1].set_ylabel('Val Acc5 (%)')
    for ax in axes:
        ax.grid(True)

    # Track whether we placed a figure-level legend (below the figure) to adjust layout
    has_fig_legend = False

    # per-architecture legend placement with optional swap
    if plotted_handles:
        ncol = max(1, len(plotted_handles) // 2)
        if not switch_legend:
            # original behavior: architectures legend below center
            fig.legend(plotted_handles, plotted_labels, loc='lower center', ncol=ncol,
                       bbox_to_anchor=(0.5, -0.06), fontsize=9, title="Architectures")
            has_fig_legend = True
        else:
            # swapped behavior: put per-architecture legend on axes[0] lower right
            arch_legend = axes[0].legend(plotted_handles, plotted_labels,
                                         title="Architectures", loc='lower right', fontsize=9)
            axes[0].add_artist(arch_legend)

    # architecture family legend created from categories_full so colors remain stable
    if not hide_family_legend and categories_full:
        cat_handles = [Line2D([0], [0], color=cat_colors[cat], lw=3) for cat in categories_full]
        if not switch_legend:
            # original: family legend on axes[0] lower right
            cat_legend = axes[0].legend(cat_handles, categories_full, title="Architecture family",
                                        loc='lower right', fontsize=9)
            axes[0].add_artist(cat_legend)
        else:
            # swapped: family legend below the figure
            ncol_cat = max(1, len(cat_handles) // 2)
            fig.legend(cat_handles, categories_full, title="Architecture family",
                       loc='lower center', ncol=ncol_cat, bbox_to_anchor=(0.5, -0.02), fontsize=9)
            has_fig_legend = True

    # adjust spacing: leave more bottom room if we used figure-level legend(s)
    bottom_rect = 0.18 if has_fig_legend else 0.08
    fig.suptitle(title, fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0, bottom_rect, 1, 1])
    plt.show()


def plot_metric(epoch_dir, arch2cat, suffix, metric, agg="mean",
                 kind="scatter", scale=1, title ="", ymin=0, top_pad=0.15,
                 loc="upper right", ax=None, figsize=(14,6)):
    arch_list = list(arch2cat.keys())
    categories = sorted(set(arch2cat.values()))
    cat_colors = {cat: f"C{i}" for i, cat in enumerate(categories)}

    vals_d = {}
    for arch in arch_list:
        df = _load_metrics_csv(os.path.join(epoch_dir, f"{arch}_{suffix}.csv"))
        if metric in df.columns:
            vals_d[arch] = getattr(df[metric], agg)() * scale

    order = sorted(vals_d.keys(), key=lambda a: vals_d[a], reverse=kind == "bar")
    vals  = [vals_d[a] for a in order]
    cols  = [cat_colors[arch2cat[a]] for a in order]

    show_fig = False
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        show_fig = True

    labels = [_fmt_arch_label(a) for a in order]

    if kind == "bar":
        bars = ax.bar(range(len(order)), vals, color=cols)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        x = np.arange(len(order))
        ax.scatter(x, vals, c=cols, s=70)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylim(ymin, max(vals) * (1 + top_pad))
    y0, y1 = ax.get_ylim()
    y_offset = (y1 - y0) * 0.02

    for idx, v in enumerate(vals):
        txt = f"{v:.1f}" if "time" in metric.lower() else f"{v:.1f}"
        if kind == "bar":
            b = bars[idx]
            ax.text(b.get_x() + b.get_width() / 2, v + y_offset, txt,
                    ha="center", va="bottom", fontsize=12)
        else:
            xi = x[idx]
            ax.text(xi, v + y_offset, txt, ha="center", va="bottom", fontsize=11)

    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    cat_handles = [Line2D([0], [0], color=cat_colors[c], lw=6) for c in categories]
    if loc is not None:
        ax.legend(cat_handles, categories, title="Architecture family", loc=loc)

    if show_fig:
        plt.tight_layout()
        plt.show()


def plot_last_epoch_hist(epoch_dir, arch2cat, suffix, metric="val_acc1",
                         title="", ymin=0, top_pad=0.15, loc="upper right", ax=None, figsize=(14,6)):
    plot_metric(epoch_dir, arch2cat, suffix, metric, agg="max",
                 kind="bar", scale=100, title = title,
                 ymin=ymin, top_pad=top_pad, loc=loc, ax=ax, figsize=figsize)


def plot_time_scatter(epoch_dir, arch2cat, suffix, metric="train_images_per_sec",
                      title="", ymin=0, top_pad=0.15, loc="upper left", ax=None, figsize=(14,6)):
    plot_metric(epoch_dir, arch2cat, suffix, metric, agg="mean",
                 kind="scatter", scale=1, title=title,
                 ymin=ymin, top_pad=top_pad, loc=loc, ax=ax, figsize=figsize)


# -----
# Grouped Bar plots
# ----

def _strip_final_b(arch):
    m = re.match(r"(.+)_b(\d+)$", arch)
    if not m:
        return arch
    base, n = m.group(1), m.group(2)
    if base.startswith("efficientnet") and n in ("0", "7"):
        return arch
    return base

def _find_key_for_arch(a, arch_keys):
    # Prefer exact match, then prefix matches, then stripped matches.
    if a in arch_keys:
        return a
    for k in arch_keys:
        if a == k or a.startswith(k + "_") or a.startswith(k):
            return k
    a_stripped = _strip_final_b(a)
    for k in arch_keys:
        if k == a_stripped or k.startswith(a_stripped + "_") or a_stripped.startswith(k):
            return k
    return a

def _compute_family_order(arch_list, arch_values, arch2cat):
    """
    arch_values: dict arch -> scalar (mean) or np.nan
    Returns ordered_archs sorted by family average (lowest->highest) then by arch value (lowest->highest).
    Families or architectures with only NaN are placed at the end.
    """
    # Group architectures by family
    family_to_archs = {}
    for arch in arch_list:
        fam = arch2cat.get(arch, None)
        family_to_archs.setdefault(fam, []).append(arch)

    # Family averages
    family_avgs = {}
    for fam, archs in family_to_archs.items():
        vals = [arch_values.get(a, np.nan) for a in archs]
        vals = [v for v in vals if not np.isnan(v)]
        family_avgs[fam] = np.nan if len(vals) == 0 else float(np.mean(vals))

    # Sort families by avg (lowest -> highest). NaN families go last.
    sorted_families = sorted(
        family_avgs.keys(),
        key=lambda f: (np.isnan(family_avgs[f]), family_avgs[f] if not np.isnan(family_avgs[f]) else 0.0)
    )

    # Within each family, sort architectures by their arch_values (lowest -> highest), NaNs last
    ordered_archs = []
    for fam in sorted_families:
        archs = family_to_archs.get(fam, [])
        archs_sorted = sorted(
            archs,
            key=lambda a: (np.isnan(arch_values.get(a, np.nan)),
                           arch_values.get(a, 0.0) if not np.isnan(arch_values.get(a, np.nan)) else 0.0)
        )
        ordered_archs.extend(archs_sorted)

    return ordered_archs

def _make_group_color_funcs(k):
    """
    Return a list of color-variant functions for k groups.
    First group -> lighten, second -> darken, subsequent -> progressively darker.
    Each returned element is a callable that accepts an rgb tuple and returns rgb tuple.
    """
    funcs = []
    for gi in range(k):
        if gi == 0:
            def fn(rgb, amt=0.22):
                r, g, b = rgb
                return (r + (1.0 - r) * amt, g + (1.0 - g) * amt, b + (1.0 - b) * amt)
            funcs.append(fn)
        elif gi == 1:
            def fn(rgb, amt=0.16):
                r, g, b = rgb
                return (r * (1.0 - amt), g * (1.0 - amt), b * (1.0 - amt))
            funcs.append(fn)
        else:
            amt = 0.12 + 0.04 * (gi - 1)
            def make_fn(amt):
                def fn(rgb):
                    r, g, b = rgb
                    return (r * (1.0 - amt), g * (1.0 - amt), b * (1.0 - amt))
                return fn
            funcs.append(make_fn(amt))
    return funcs

def _draw_grouped_bars(ax, ordered_items, groups_values, base_colors, group_color_funcs,
                       xtick_labels, value_fmt="{:.1f}", ylabel="", title="", ymin=0, top_pad=0.15):
    """
    Draw grouped bars on ax.
    - ordered_items: list of items (architectures)
    - groups_values: list of lists; len == number of groups; each inner list length == len(ordered_items)
    - base_colors: list of base color strings (one per ordered_item)
    - group_color_funcs: list of callables to create variant rgb from base rgb
    - xtick_labels: labels to show on x axis (same order as ordered_items)
    - value_fmt: format string for annotations
    """
    n = len(ordered_items)
    k = len(groups_values)
    x = np.arange(n)
    total_width = 0.8
    bar_width = total_width / k
    offsets = (np.arange(k) - (k - 1) / 2.0) * bar_width

    # prepare facecolors for each group and item
    bars_by_group = []
    for gi in range(k):
        vals = groups_values[gi]
        xs = x + offsets[gi]
        heights = []
        facecolors = []
        for idx, arch in enumerate(ordered_items):
            raw_color = base_colors[idx]
            rgb = mcolors.to_rgb(raw_color)
            rgb_variant = group_color_funcs[gi](rgb)
            val = vals[idx]
            if np.isnan(val):
                heights.append(0.0)
                facecolors.append("none")
            else:
                heights.append(val)
                facecolors.append(mcolors.to_hex(rgb_variant))
        bars = ax.bar(xs, heights, width=bar_width, align="center",
                      color=facecolors, edgecolor="white", linewidth=1.2)
        bars_by_group.append(bars)

    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # y limits
    all_vals = np.array([v for grp in groups_values for v in grp], dtype=float)
    finite_vals = all_vals[np.isfinite(all_vals)]
    ymax = (finite_vals.max() if finite_vals.size else 1.0) * (1 + top_pad)
    ax.set_ylim(ymin, ymax)

    # annotate
    for bars in bars_by_group:
        for rect in bars:
            height = rect.get_height()
            fc = rect.get_facecolor()
            if isinstance(fc, str) and fc == 'none':
                continue
            if isinstance(fc, tuple) and len(fc) == 4 and fc[3] == 0:
                continue
            if height and height > 0:
                txt = value_fmt.format(height)
                ax.text(rect.get_x() + rect.get_width() / 2,
                        height + (ymax - ymin) * 0.02,
                        txt, ha="center", va="bottom", fontsize=10)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

def plot_grouped_bars(epoch_dir, arch2cat, suffixes, suffix_labels=None, metric="train_time_sec", agg="mean", scale=1.0, title="", 
                      ymin=0, top_pad=0.15, figsize=(14, 8), ax=None, arch_df=None, arch_cols=("FLOPs_G", "params_M"), arch_col_labels=None):
    """
    Grouped bar plot for multiple suffixes (runs) or, when metric == "model_size" and
    arch_df is provided, plot two columns from arch_df per architecture.

    - If metric == "model_size" and arch_df is provided, arch_df is expected to contain
      an 'arch' column and the two columns in `arch_cols`. The function will behave like
      the former plot_arch_grouped_bars for those two columns.
    - Otherwise the function reads CSVs / model files under epoch_dir using `suffixes`
      and the existing metric/agg logic (including the file-size special-case when
      metric == "file_size").
    """
    if suffix_labels is None:
        suffix_labels = suffixes
    if arch_col_labels is None:
        arch_col_labels = list(arch_cols)

    # category colors
    arch_list = list(arch2cat.keys())
    categories = sorted(set(arch2cat.values()))
    cat_colors = {cat: f"C{i}" for i, cat in enumerate(categories)}
    arch_keys = list(arch2cat.keys())

    # ---- Branch: plot from arch_df (two-column table) ----
    if metric == "model_size" and arch_df is not None:
        # determine ordering (family ordering by average of the two arch_cols)
        archs_in_df = list(arch_df["arch"])
        arch_to_key = {a: _find_key_for_arch(a, arch_keys) for a in archs_in_df}

        arch_mean_values = {}
        for a in archs_in_df:
            v0 = arch_df.loc[arch_df["arch"] == a, arch_cols[0]].values
            v1 = arch_df.loc[arch_df["arch"] == a, arch_cols[1]].values
            vals_x = []
            if v0.size and not pd.isna(v0[0]):
                vals_x.append(float(v0[0]))
            if v1.size and not pd.isna(v1[0]):
                vals_x.append(float(v1[0]))
            arch_mean_values[a] = np.nan if len(vals_x) == 0 else float(np.mean(vals_x))

        order = _compute_family_order(archs_in_df, arch_mean_values,
                                      {a: arch2cat.get(_find_key_for_arch(a, arch_keys), None) for a in archs_in_df})

        # gather values and base colors for the two arch_cols
        vals0 = []
        vals1 = []
        base_colors = []
        xtick_labels = []
        for a in order:
            k = _find_key_for_arch(a, arch_keys)
            cat = arch2cat.get(k, arch2cat.get(a, None))
            if cat not in cat_colors:
                cat_colors[cat] = f"C{len(cat_colors)}"
            base_colors.append(cat_colors[cat])
            xtick_labels.append(_strip_final_b(k))
            v0 = arch_df.loc[arch_df["arch"] == a, arch_cols[0]].values
            v1 = arch_df.loc[arch_df["arch"] == a, arch_cols[1]].values
            vals0.append(float(v0[0]) if v0.size and not pd.isna(v0[0]) else np.nan)
            vals1.append(float(v1[0]) if v1.size and not pd.isna(v1[0]) else np.nan)

        groups_values = [vals0, vals1]
        group_color_funcs = _make_group_color_funcs(2)

        show_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            show_fig = True

        _draw_grouped_bars(ax, order, groups_values, base_colors, group_color_funcs,
                           xtick_labels, value_fmt="{:.1f}", ylabel=" / ".join(arch_cols),
                           title=title, ymin=ymin, top_pad=top_pad)

        # legends
        cat_handles = [
            Patch(facecolor=mcolors.to_hex(mcolors.to_rgb(cat_colors[c])), edgecolor="white", label=c)
            for c in sorted(cat_colors.keys())
        ]
        sample_cat_rgb = mcolors.to_rgb(next(iter(cat_colors.values())))
        g0 = mcolors.to_hex(group_color_funcs[0](sample_cat_rgb))
        g1 = mcolors.to_hex(group_color_funcs[1](sample_cat_rgb))
        group_handles = [
            Patch(facecolor=g0, edgecolor="white", label=arch_col_labels[0]),
            Patch(facecolor=g1, edgecolor="white", label=arch_col_labels[1]),
        ]

        leg1 = ax.legend(handles=cat_handles, title="Architecture family",
                         loc="upper left", bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes)
        ax.add_artist(leg1)
        ax.legend(handles=group_handles, title="Metric",
                  loc="upper left", bbox_to_anchor=(0.30, 0.98), bbox_transform=ax.transAxes)

        plt.tight_layout()
        if show_fig:
            plt.show()
        return ax

    # ---- Else: existing suffix / CSV / file-size behavior ----

    # collect values
    vals = {arch: [np.nan] * len(suffixes) for arch in arch_list}
    for i, suf in enumerate(suffixes):
        for arch in arch_list:
            if metric == "file_size":
                path = os.path.join(epoch_dir, f"{arch}_{suf}.pth")
                try:
                    size_bytes = os.path.getsize(path) / 1024**2
                except (OSError, FileNotFoundError):
                    size_bytes = None
                vals[arch][i] = float(size_bytes) * scale if size_bytes is not None else np.nan
            else:
                path = os.path.join(epoch_dir, f"{arch}_{suf}.csv")
                try:
                    df = _load_metrics_csv(path)
                except Exception:
                    vals[arch][i] = np.nan
                    continue
                if metric not in df.columns:
                    vals[arch][i] = np.nan
                    continue
                try:
                    agg_val = getattr(df[metric], agg)()
                except Exception:
                    vals[arch][i] = np.nan
                    continue
                vals[arch][i] = float(agg_val) * scale if pd.notna(agg_val) else np.nan

    # per-architecture mean
    arch_means = {}
    for arch, arr in vals.items():
        a = np.array(arr, dtype=float)
        arch_means[arch] = np.nanmean(a) if not np.all(np.isnan(a)) else np.nan

    # determine order by family averages
    ordered_archs = _compute_family_order(arch_list, arch_means, arch2cat)

    # prepare groups_values and base colors
    groups_values = []
    for gi in range(len(suffixes)):
        groups_values.append([vals[arch][gi] for arch in ordered_archs])

    base_colors = [cat_colors[arch2cat[arch]] for arch in ordered_archs]
    xtick_labels = [_fmt_arch_label(a) for a in ordered_archs]

    show_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_fig = True

    group_color_funcs = _make_group_color_funcs(len(suffixes))
    _draw_grouped_bars(ax, ordered_archs, groups_values, base_colors, group_color_funcs,
                       xtick_labels, value_fmt="{:.0f}", ylabel=metric, title=title,
                       ymin=ymin, top_pad=top_pad)

    # legends
    cat_handles = [
        Patch(facecolor=mcolors.to_hex(mcolors.to_rgb(cat_colors[c])), edgecolor="white", label=c)
        for c in categories
    ]
    sample_cat_rgb = mcolors.to_rgb(cat_colors[categories[0]])
    group_handles = []
    for gi in range(len(suffixes)):
        group_handles.append(Patch(facecolor=mcolors.to_hex(group_color_funcs[gi](sample_cat_rgb)),
                                  edgecolor="white", label=suffix_labels[gi]))

    leg1 = ax.legend(handles=cat_handles, title="Architecture family",
                     loc="upper left", bbox_to_anchor=(0.02, 0.98), bbox_transform=ax.transAxes)
    ax.add_artist(leg1)
    ax.legend(handles=group_handles, title="Run",
              loc="upper left", bbox_to_anchor=(0.20, 0.98), bbox_transform=ax.transAxes)

    plt.tight_layout()
    if show_fig:
        plt.show()
    return ax


def plot_per_class_metric(df, label_col='label', metric_col='accuracy', title = "",ascending=True, mean_std_mult=1.0, scale = 100, 
                          low_color="#D64550", high_color="#2ECC71", neutral_color=None, figsize=(12, None), legend_loc='lower left'):
    "Plot horizontal bars for a per-class metric and auto-annotate under/overperformers."
    df_sorted = df.sort_values(metric_col, ascending=ascending).reset_index(drop=True)
    df_sorted[metric_col] = df_sorted[metric_col] * scale
    n = len(df_sorted)

    mean_val = float(df_sorted[metric_col].mean())
    std_val = float(df_sorted[metric_col].std(ddof=0))
    low_thr = mean_val - mean_std_mult * std_val
    high_thr = mean_val + mean_std_mult * std_val

    if neutral_color is None:
        neutral_color = sns.color_palette("deep")[0]

    colors = [low_color if v < low_thr else (high_color if v >= high_thr else neutral_color)
              for v in df_sorted[metric_col].values]

    width, height = figsize
    fig_height = (max(6, 0.12 * n) if height is None else height)
    bar_height = 0.45

    fig, ax = plt.subplots(figsize=(width, fig_height))
    y_pos = np.arange(n)

    ax.barh(y_pos, df_sorted[metric_col].values, height=bar_height,
            color=colors, edgecolor="#12303f", linewidth=0.5, alpha=0.95)

    ax.axvline(mean_val, color="#ffd166", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.text(mean_val + 0.005, -1, f"Mean = {mean_val:.1f}", color="#ffd166", va="center", fontsize=10)

    # Annotate all underperforming and overperforming entries
    under_idx = df_sorted.index[df_sorted[metric_col] < low_thr].tolist()
    over_idx = df_sorted.index[df_sorted[metric_col] >= high_thr].tolist()

    for i in under_idx:
        val = df_sorted[metric_col].iloc[i]
        ax.text(val + 0.005, i, f"{val:.1f}", va="center", ha="left", fontsize=8, color="#ffd6d6")

    for i in over_idx:
        val = df_sorted[metric_col].iloc[i]
        ax.text(val + 0.005, i, f"{val:.1f}", va="center", ha="left", fontsize=8, color="#bfe3b4")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted[label_col].values, fontsize=9)
    ax.set_xlabel(metric_col.replace('_', ' ').title(), fontsize=12)
    ax.set_xlim(0, max(1.0, df_sorted[metric_col].max() * 1.05))
    title_order = "ascending" if ascending else "descending"
    ax.set_title(title, fontsize=18, pad=12)

    ax.xaxis.grid(True, color="#2a3f5f", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.yaxis.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_elems = [
        Patch(facecolor=low_color, edgecolor="#12303f", label=f"Underperforming (< {low_thr:.1f})"),
        Patch(facecolor=neutral_color, edgecolor="#12303f", label="Around average"),
        Patch(facecolor=high_color, edgecolor="#12303f", label=f"Overperforming (>= {high_thr:.1f})"),
    ]
    ax.legend(handles=legend_elems, loc=legend_loc, frameon=True, facecolor="#0d1b2a", edgecolor="#444444", fontsize=9)

    plt.tight_layout()
    return fig, ax





def plot_top_confusion_submatrix(y_true, y_pred, classes, k=20, figsize=(12,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    err = cm.copy()
    np.fill_diagonal(err, 0)

    topk = np.argpartition(err.ravel(), -k)[-k:]
    r, c = np.unravel_index(topk, err.shape)
    order = np.argsort(err[r, c])[::-1]
    r, c = r[order], c[order]

    focus = np.unique(np.concatenate([r, c]))
    sub = cm[np.ix_(focus, focus)]
    sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=figsize)
    plt.imshow(sub, aspect="auto")
    plt.xticks(range(len(focus)), [classes[i] for i in focus], rotation=90)
    plt.yticks(range(len(focus)), [classes[i] for i in focus])
    plt.title("Confusion submatrix from top off-diagonal confusions (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def top_confusions_table(y_true, y_pred, classes, df_metrics, k=20):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    support = df_metrics.set_index("label").loc[classes, "support"].to_numpy()

    i, j = np.where(off > 0)
    counts = off[i, j]
    order = np.argsort(counts)[-k:][::-1]
    pct = counts / support[i]

    df = pd.DataFrame({
        "true": [classes[a] for a in i[order]],
        "pred": [classes[b] for b in j[order]],
        "count": counts[order],
        "pct_of_true": pct[order],
    })

    return (df.style
              .format({"pct_of_true": "{:.2%}"}, na_rep="")
              .hide(axis="index")
              .bar(subset=["count"], vmin=0)
              .background_gradient(subset=["pct_of_true"]))
    

def build_confusion_grid(images_dir, test_txt, y_pred, y_score, labels, ext=".jpg"):
    """Return (selected_paths, titles) for a confusion-style image grid."""
    classes = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(test_txt, "r", encoding="utf-8") as f:
        test_lines = [ln.strip() for ln in f if ln.strip()]
    image_paths = [os.path.join(images_dir, entry + ext) for entry in test_lines]

    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    true_indices = np.array([class_to_idx[entry.split("/")[0]] for entry in test_lines])

    def pick_index(true_idx, pred_idx):
        exact = np.where((true_indices == true_idx) & (y_pred == pred_idx))[0]
        if exact.size:
            return int(exact[0])
        cand = np.where(true_indices == true_idx)[0]
        if cand.size:
            return int(cand[0])
        return int(np.argmax(y_score[:, pred_idx]))

    selected_paths = []
    titles = []
    for t_label in labels:
        t_idx = class_to_idx[t_label]
        for p_label in labels:
            p_idx = class_to_idx[p_label]
            i = pick_index(t_idx, p_idx)
            selected_paths.append(image_paths[i])

            top3 = np.argsort(y_score[i])[::-1][:3]
            preds = [classes[idx] for idx in top3]
            scores = (y_score[i, top3] * 100.0)

            title = (
                f"Label : {t_label}\n"
                f"Prediction 1 : {preds[0]} {scores[0]:.1f}%\n"
                f"Prediction 2 : {preds[1]} {scores[1]:.1f}%\n"
                f"Prediction 3 : {preds[2]} {scores[2]:.1f}%"
            )
            titles.append(title)

    return selected_paths, titles

def plot_calibration_and_histogram(y_score, y_true ,figsize=(12, 4), n_bins = 20):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    confidences = np.max(y_score, axis=1)
    predictions = np.argmax(y_score, axis=1)
    correctness = (predictions == y_true).astype(int)
    prob_true, prob_pred = calibration_curve(correctness, confidences, n_bins=n_bins)
    axs[0].plot(prob_pred, prob_true, marker='o', linewidth=1)
    axs[0].plot([0, 1], [0, 1], color='w', alpha=0.5, linestyle='--')
    axs[0].set_xlabel('Mean predicted probability')
    axs[0].set_ylabel('Fraction of positives')
    axs[0].set_title('Reliability diagram')

    axs[1].hist(confidences, bins=n_bins)
    axs[1].set_xlabel('Predicted max confidence')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Confidence histogram')
    plt.tight_layout()
    plt.show()
    
