import numpy as np
import matplotlib.pyplot as plt
import textwrap
from typing import List, Tuple, Optional, Dict, Union
import time
import os
import warnings
import seaborn as sns
from ..data.csr import get_degree_frequency_csr, get_nelements_by_row_csr, get_nrows_csr

try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False

BW_MARKERS = ['o', 's', '^', 'D', 'x', '*', 'v', 'P']
BW_LINESTYLES = ['-', '--', '-.', ':']

def save_plot(
    path,
    fig=None,
    formats="pdf",
    timestamps=False,
    use_tex=False,
    overwrite=False,
    dpi=300,
):
    old_usetex = plt.rcParams.get("text.usetex", False)

    if use_tex:
        try:
            plt.rcParams["text.usetex"] = True
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                (fig or plt.gcf()).canvas.draw()
        except Exception:
            plt.rcParams["text.usetex"] = False

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    base, _ = os.path.splitext(path)

    if formats == "all":
        formats = ["pdf", "png", "svg"]
    elif isinstance(formats, str):
        formats = [formats]

    timestamp = time.strftime("%Y%m%d_%H%M%S") if timestamps else ""

    saved = []

    for fmt in formats:
        final = f"{base}_{timestamp}.{fmt}" if timestamp else f"{base}.{fmt}"

        if not overwrite and os.path.exists(final):
            stem, ext = os.path.splitext(final)
            i = 1
            alt = f"{stem}_{i}{ext}"
            while os.path.exists(alt):
                i += 1
                alt = f"{stem}_{i}{ext}"
            final = alt

        fig.savefig(final, dpi=dpi, bbox_inches="tight")
        saved.append(final)

    plt.rcParams["text.usetex"] = old_usetex

    return saved

def plot_ratings_csr_publication(
    ratings_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    label: str = None,
    title: str = None,
    log_plot: bool = True,
    bw_plot: bool = True,
    color: str = None,
    figsize: Tuple[int, int] = (7, 5),
    fig_ax: Tuple = None,
    hist: bool = False,
    bins: int = 1000
) -> Tuple:
    """
    Plot rating-degree distributions for CSR matrices.
    
    Args:
        ratings_csr: CSR matrix tuple.
        label: Legend label.
        title: Plot title.
        log_plot: Whether to use log scale.
        bw_plot: Whether to use black and white colors.
        color: Plot color.
        figsize: Figure size.
        fig_ax: Existing (fig, ax) tuple.
        hist: If True, plot histogram; if False, plot scatter.
        bins: Number of bins for histogram.
    
    Returns:
        Tuple of (fig, ax).
    """
    x_degree, y_frequency = get_degree_frequency_csr(ratings_csr)

    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)

    if bw_plot:
        color = "black"

    if hist is True:
        expanded = np.repeat(x_degree, y_frequency)
        if bw_plot:
            color = 'lightgray'
        ax.hist(
            expanded,
            bins=bins,
            color=color,
            alpha=0.7,
            edgecolor="black",
            label=label,
        )

        if log_plot:
            ax.set_xscale("log")

        ax.set_xlabel("number of Ratings")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(False)
        plt.tight_layout()
        return fig, ax

    ax.scatter(x_degree, y_frequency, s=2, alpha=0.7, color=color, label=label)

    if log_plot:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()

    return fig, ax


def plot_bw_scatter_lines(
    data_list: List[Tuple[np.ndarray, np.ndarray]],
    labels: List[str] = None,
    log_scale: bool = False,
    title: str = "BW Plot",
    figsize: Tuple[int, int] = (7, 5),
    fig_ax: Tuple = None
) -> Tuple:
    """
    Plot multiple series in black-and-white using different markers and line styles.
    
    Args:
        data_list: List of (x, y) data tuples.
        labels: List of labels for each series.
        log_scale: Whether to use log scale.
        title: Plot title.
        figsize: Figure size.
        fig_ax: Existing (fig, ax) tuple.
    
    Returns:
        Tuple of (fig, ax).
    """
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)

    n_series = len(data_list)
    for i, (x, y) in enumerate(data_list):
        marker = BW_MARKERS[i % len(BW_MARKERS)]
        linestyle = BW_LINESTYLES[i % len(BW_LINESTYLES)]
        label = labels[i] if labels else None

        ax.plot(x, y, marker=marker, linestyle=linestyle, color='black', alpha=0.8, label=label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(False)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if labels:
        ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_als_curve_only(
    train_hist: np.ndarray,
    test_hist: np.ndarray,
    title: str = "",
    ylabel: str = "",
    suptitle: str = None,
    fig_ax: Tuple = None,
    figsize: Tuple[int, int] = (6, 4)
) -> Tuple:
    """
    Plot single training curve (loss or RMSE).
    
    Args:
        train_hist: Training history.
        test_hist: Test history.
        title: Plot title.
        ylabel: Y-axis label.
        suptitle: Figure suptitle.
        fig_ax: Existing (fig, ax) tuple.
        figsize: Figure size.
    
    Returns:
        Tuple of (fig, ax).
    """
    epoch_range = range(1, len(train_hist) + 1)
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epoch_range, train_hist, marker="", label="train")
    ax.plot(epoch_range, test_hist, marker="", ls='--', label="test")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, ax


def plot_als_curves(
    train_loss_hist: np.ndarray,
    test_loss_hist: np.ndarray,
    train_rmse_hist: np.ndarray,
    test_rmse_hist: np.ndarray,
    K: int,
    precision: float,
    vector_penality: float,
    bias_penality: float,
    bias_only: bool = False,
    fig_ax: Tuple = None,
    toplot: str = "both",
    with_suptitle: bool = True
) -> Tuple:
    """
    Plot ALS training curves (loss and/or RMSE).
    
    Args:
        train_loss_hist: Training loss history.
        test_loss_hist: Test loss history.
        train_rmse_hist: Training RMSE history.
        test_rmse_hist: Test RMSE history.
        K: Embedding dimension.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        bias_only: Whether this is a bias-only model.
        fig_ax: Existing (fig, ax) tuple.
        toplot: What to plot ('both', 'loss', or 'rmse').
        with_suptitle: Whether to include suptitle.
    
    Returns:
        Tuple of (fig, ax) or (fig, axes).
    """
    plot_loss = False
    plot_rmse = False
    
    if toplot == "both":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        ax_loss = axes[0]
        ax_rmse = axes[1]
        plot_loss = True
        plot_rmse = True
        fig_ax = fig, axes
    else:
        if fig_ax:
            fig, ax = fig_ax
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig_ax = fig, ax
        
        if toplot == "loss":
            plot_loss = True
            ax_loss = ax
        elif toplot == "rmse":
            plot_rmse = True
            ax_rmse = ax
    
    if plot_loss:
        title = "Loss"
        ylabel = "loss"
        plot_als_curve_only(
            train_loss_hist,
            test_loss_hist,
            title=title,
            ylabel=ylabel,
            fig_ax=(fig, ax_loss),
        )
    
    if plot_rmse:
        title = "RMSE"
        ylabel = "rmse"
        plot_als_curve_only(
            train_rmse_hist,
            test_rmse_hist,
            title=title,
            ylabel=ylabel,
            fig_ax=(fig, ax_rmse),
            figsize=(6, 4)
        )

    if with_suptitle:
        if bias_only:
            fig.suptitle(fr"$K$={K}, $\lambda$={precision}, $\tau_{{bias}}$={bias_penality}")
        else:
            fig.suptitle(fr"$K$={K}, $\lambda$={precision}, $\tau$={vector_penality}, $\tau_{{bias}}$={bias_penality}")
        fig.tight_layout()
    return fig_ax


def plot_als_curves_multiple_models(
    fitted_models: List[Dict],
    fig_ax: Tuple = None,
    toplot: str = "both",
    with_suptitle: bool = True
) -> None:
    """
    Plot training curves for multiple models.
    
    Args:
        fitted_models: List of model dictionaries.
        fig_ax: Existing (fig, ax) tuple.
        toplot: What to plot ('both', 'loss', or 'rmse').
        with_suptitle: Whether to include suptitle.
    """
    for model in fitted_models:
        bias_only = len(model["user_vectors"]) == 0
        
        train_loss_hist = model["train_loss_hist"]
        test_loss_hist = model["test_loss_hist"]
        train_rmse_hist = model["train_rmse_hist"]
        test_rmse_hist = model["test_rmse_hist"]
        precision = model["precision"]
        vector_penality = model["vector_penality"]
        bias_penality = model["bias_penality"]
        K = model["K"]
        
        fig, ax = plot_als_curves(
            train_loss_hist,
            test_loss_hist,
            train_rmse_hist,
            test_rmse_hist,
            K,
            precision,
            vector_penality,
            bias_penality,
            bias_only=bias_only,
            fig_ax=fig_ax,
            toplot=toplot,
            with_suptitle=with_suptitle)
    



def plot_bar_general(
    x_labels: List[str],
    y_values: np.ndarray,
    sort: bool = True,
    topn: int = None,
    bw_plot: bool = True,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Bar Plot",
    rotate_xticks: int = 45,
    horizontal: bool = False,
    max_height_per_label: float = 0.5,
    max_label_width: int = 20,
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    General bar plot function with flexible options.

    Args:
        x_labels: Labels for bars.
        y_values: Values for bars.
        sort: Whether to sort by value.
        topn: Show only top N items.
        bw_plot: Use black and white colors.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        rotate_xticks: Rotation angle for x-tick labels.
        horizontal: If True, create horizontal bar plot.
        max_height_per_label: Height per bar for auto-sizing.
        max_label_width: Maximum characters per line in labels.
        fig_ax: Optional (fig, ax). If None, a new figure is created.

    Returns:
        (fig, ax)
    """
    y_values = np.asarray(y_values)

    indices = np.arange(len(x_labels))
    if sort:
        indices = np.flip(np.argsort(y_values))
    if topn is not None:
        indices = indices[:topn]

    x_labels_sorted = [x_labels[i] for i in indices]
    y_values_sorted = y_values[indices]

    wrapped_labels = [
        "\n".join(textwrap.wrap(label, max_label_width))
        for label in x_labels_sorted
    ]

    n_bars = len(wrapped_labels)

    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()

        if horizontal:
            fig_height = max(5, n_bars * max_height_per_label)
            fig.set_size_inches(10, fig_height)
        else:
            fig_width = max(8, n_bars * max_height_per_label)
            fig.set_size_inches(fig_width, 5)

    bar_color = "lightgray" if bw_plot else "C0"
    edge_color = "black"

    if horizontal:
        ax.barh(range(n_bars), y_values_sorted,
                color=bar_color, edgecolor=edge_color)
        ax.set_yticks(range(n_bars))
        ax.set_yticklabels(wrapped_labels)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ax.invert_yaxis()
    else:
        ax.bar(range(n_bars), y_values_sorted,
               color=bar_color, edgecolor=edge_color)
        ax.set_xticks(range(n_bars))
        ax.set_xticklabels(
            wrapped_labels,
            rotation=rotate_xticks,
            ha="right"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()

    for label, value in zip(x_labels_sorted, y_values_sorted):
        print(fr"{label}: & {value} \\")

    return fig, ax

def plot_box_general(
    y_values: List[float],
    label: str = "",
    bw_plot: bool = True,
    ylabel: str = "Y",
    title: str = "Box Plot",
    horizontal: bool = False,
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single box from scalar values.

    Args:
        y_values: List of scalar values.
        label: Label for the box.
        bw_plot: Use black and white style.
        ylabel: Y-axis label.
        title: Plot title.
        horizontal: If True, draw a horizontal box plot.
        fig_ax: Optional (fig, ax).

    Returns:
        (fig, ax)
    """
    y = np.asarray(y_values, dtype=float)
    y = y[~np.isnan(y)]

    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    boxprops = dict(facecolor="lightgray", edgecolor="black") if bw_plot else {}
    medianprops = dict(color="black")
    whiskerprops = dict(color="black")
    capprops = dict(color="black")

    ax.boxplot(
        [y],
        vert=not horizontal,
        patch_artist=bw_plot,
        labels=[label] if not horizontal else None,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )

    if horizontal:
        ax.set_yticks([1])
        ax.set_yticklabels([label])
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
    else:
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()

    return fig, ax

def plot_multiple_box_general(
    x_labels: List[str],
    y_values: List[List[float]],
    sort: bool = True,
    topn: int = None,
    bw_plot: bool = True,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Box Plot",
    rotate_xticks: int = 45,
    horizontal: bool = False,
    max_height_per_label: float = 0.5,
    max_label_width: int = 20,
    sort_stat: str = "median",  # "median" or "mean"
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    General box plot function with flexible options.

    Args:
        x_labels: Labels for boxes.
        y_values: Iterable of arrays/lists (one per label).
        sort: Whether to sort boxes by a statistic.
        topn: Show only top N items.
        bw_plot: Use black and white style.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        rotate_xticks: Rotation angle for x-tick labels.
        horizontal: If True, create horizontal box plot.
        max_height_per_label: Height per box for auto-sizing.
        max_label_width: Maximum characters per line in labels.
        sort_stat: Statistic for sorting ("median" or "mean").
        fig_ax: Optional (fig, ax). If None, a new figure is created.

    Returns:
        (fig, ax)
    """
    y_values = [np.asarray(v, dtype=float) for v in y_values]

    stats = []
    for v in y_values:
        v = v[~np.isnan(v)]
        if sort_stat == "mean":
            stats.append(np.mean(v) if len(v) else -np.inf)
        else:
            stats.append(np.median(v) if len(v) else -np.inf)
    stats = np.asarray(stats)

    indices = np.arange(len(x_labels))
    if sort:
        indices = np.flip(np.argsort(stats))
    if topn is not None:
        indices = indices[:topn]

    x_labels_sorted = [str(x_labels[i]) for i in indices]
    y_values_sorted = [y_values[i] for i in indices]

    wrapped_labels = [
        "\n".join(textwrap.wrap(label, max_label_width))
        for label in x_labels_sorted
    ]

    n_boxes = len(wrapped_labels)

    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()

        if horizontal:
            fig_height = max(5, n_boxes * max_height_per_label)
            fig.set_size_inches(10, fig_height)
        else:
            fig_width = max(8, n_boxes * max_height_per_label)
            fig.set_size_inches(fig_width, 5)

    boxprops = dict(facecolor="lightgray", edgecolor="black") if bw_plot else {}
    medianprops = dict(color="black")
    whiskerprops = dict(color="black")
    capprops = dict(color="black")

    ax.boxplot(
        y_values_sorted,
        vert=not horizontal,
        patch_artist=bw_plot,
        labels=wrapped_labels if not horizontal else None,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )

    if horizontal:
        ax.set_yticks(np.arange(1, n_boxes + 1))
        ax.set_yticklabels(wrapped_labels)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ax.invert_yaxis()
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(
            wrapped_labels,
            rotation=rotate_xticks,
            ha="right"
        )

    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()

    return fig, ax


def plot_distribution_general(
    x_labels: List[str],
    y_values: List[List[float]],
    min: float = 0.0,
    max: float = 5.0,
    bin_width: float = 0.5,
    sort: bool = False,
    topn: Optional[int] = None,
    sort_stat: str = "median",
    bw_plot: bool = True,
    xlabel: str = "Rating",
    ylabel: str = "Density",
    title: str = "Distribution Plot",
    horizontal: bool = False,
    rotate_xticks: int = 0,
    max_height_per_label: float = 0.5,   # API consistency
    max_label_width: int = 20,
    fig_ax: Optional[Tuple[plt.Figure, np.ndarray]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot per-label rating distributions using subplots with independent x-axes.

    Each subplot shows a histogram of ratings for a label, normalized to density.
    Tick labels for ratings are displayed for every subplot from `min` to `max` with step `bin_width`.
    Labels can be optionally sorted and truncated. Optionally, only the top-N labels are shown.

    Args:
        x_labels (List[str]): List of label names (e.g., movie titles, traits).
        y_values (List[List[float]]): Ratings corresponding to each label.
        min (float, optional): Minimum rating value. Defaults to 0.0.
        max (float, optional): Maximum rating value. Defaults to 5.0.
        bin_width (float, optional): Width of histogram bins. Defaults to 0.5.
        sort (bool, optional): Whether to sort labels by statistic. Defaults to False.
        topn (Optional[int], optional): Keep only top-N labels after sorting. Defaults to None.
        sort_stat (str, optional): Statistic for sorting ("mean", "median", "max", "min"). Defaults to "median".
        bw_plot (bool, optional): Use grey color if True; else default color. Defaults to True.
        xlabel (str, optional): X-axis label. Defaults to "Rating".
        ylabel (str, optional): Y-axis label. Defaults to "Density".
        title (str, optional): Figure title. Defaults to "Distribution Plot".
        horizontal (bool, optional): Horizontal layout if True. Defaults to False.
        rotate_xticks (int, optional): Rotation angle for x-tick labels. Defaults to 0.
        max_height_per_label (float, optional): Max height for histogram normalization. Defaults to 0.5.
        max_label_width (int, optional): Max label name width (truncate if longer). Defaults to 20.
        fig_ax (Optional[Tuple[plt.Figure, np.ndarray]], optional): Optional (fig, axes) to plot into. Defaults to None.

    Returns:
        Tuple[plt.Figure, np.ndarray]: The figure and axes array for further customization.
    """

    assert len(x_labels) == len(y_values), "x_labels and y_values must match"

    def stat_fn(v):
        v = np.asarray(v)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return np.nan
        return {
            "mean": np.mean,
            "median": np.median,
            "max": np.max,
            "min": np.min,
        }[sort_stat](v)

    stats = np.array([stat_fn(v) for v in y_values])
    order = np.arange(len(x_labels))

    if sort:
        order = np.argsort(stats)[::-1]

    if topn is not None:
        order = order[:topn]

    x_labels = [x_labels[i][:max_label_width] for i in order]
    y_values = [y_values[i] for i in order]

    n = len(x_labels)

    bins = np.arange(min, max + bin_width, bin_width)
    centers = bins[:-1] + bin_width / 2
    tick_vals = np.arange(min, max + bin_width, bin_width)
    tick_labels = [f"{t:.1f}" for t in tick_vals]

    if fig_ax is None:
        if horizontal:
            fig, axes = plt.subplots(1, n, sharex=False, sharey=True, figsize=(3.5 * n, 4))
        else:
            fig, axes = plt.subplots(n, 1, sharex=False, sharey=True, figsize=(6, 2.4 * n))
    else:
        fig, axes = fig_ax

    axes = np.atleast_1d(axes)
    bar_color = "0.6" if bw_plot else None

    for ax, label, ratings in zip(axes, x_labels, y_values):
        ratings = np.asarray(ratings)
        ratings = ratings[(ratings >= min) & (ratings <= max)]

        counts, _ = np.histogram(ratings, bins=bins)
        if counts.sum() == 0:
            continue

        density = counts / counts.sum()

        if horizontal:
            ax.barh(centers, density, height=bin_width, color=bar_color, edgecolor="0.3")
            ax.set_title(f"{label} Ratings")
            ax.set_yticks(tick_vals)
            ax.set_yticklabels(tick_labels)
            ax.set_ylim(min, max)
            ax.set_xlabel(xlabel)
        else:
            ax.bar(centers, density, width=bin_width, color=bar_color, edgecolor="0.3")
            ax.set_title(f"{label} Ratings")
            # ax.set_ylabel(f"{label}", rotation=0, labelpad=45, va="center")
            ax.set_xticks(tick_vals)
            ax.set_xticklabels(tick_labels, rotation=rotate_xticks)
            ax.set_xlim(min, max)
            ax.set_xlabel(xlabel)

        ax.grid(False)

    fig.suptitle(title)
    plt.tight_layout()

    return fig, axes

def plot_feature_distribution_bar(
    movies_by_feature_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    feature_idx_to_id: List[str],
    sort: bool = True,
    topn: int = None,
    horizontal: bool = True,
    bw_plot: bool = True,
    figsize: Tuple[int, int] = None
) -> Tuple:
    """
    Plot bar chart of movie counts per feature.
    
    Args:
        movies_by_feature_csr: Movies by feature CSR matrix.
        feature_idx_to_id: Feature index to ID mapping.
        sort: Whether to sort by count.
        topn: Show only top N features.
        horizontal: If True, create horizontal bar plot.
        bw_plot: Use black and white colors.
        figsize: Figure size.
    
    Returns:
        Tuple of (fig, ax).
    """
    nmovies_per_feature = get_nelements_by_row_csr(movies_by_feature_csr)
    indices = np.arange(len(feature_idx_to_id))
    feature_names = [feature_idx_to_id[i] for i in indices]

    fig, ax = plot_bar_general(
        feature_names,
        nmovies_per_feature,
        sort=sort,
        topn=topn,
        horizontal=horizontal,
        bw_plot=bw_plot,
        xlabel="Feature",
        ylabel="Number of Movies",
        title="Number of Movies per Feature",
        figsize=figsize
    )
    return fig, ax


def plot_metric_scatter(
    x: List,
    y: List,
    z: List,
    xlabel: str = "K",
    ylabel: str = r"$\tau$",
    title: str = "RMSE",
    fig_ax: Tuple = None
) -> Tuple:
    """
    Create scatter plot with labeled points.
    
    Args:
        x: X values (e.g., K values).
        y: Y values (e.g., tau values).
        z: Z values (e.g., RMSE values) for coloring and labels.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        fig_ax: Existing (fig, ax) tuple.
    
    Returns:
        Tuple of (fig, ax).
    """
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(x, y, c=z, cmap='viridis', s=80)

    texts = []
    for xi, yi, zi in zip(x, y, z):
        texts.append(ax.text(xi, yi, f"{zi:.3f}", fontsize=11))

    if _HAS_ADJUSTTEXT:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="->", color='gray', lw=0.7),
            force_points=0.5,
            force_text=0.5,
            expand_text=(1.2, 1.2)
        )

    ax.set_xticks(sorted(set(x)))
    ax.set_yticks(sorted(set(y)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.colorbar(scatter, ax=ax, label="RMSE")

    ax.grid(False)
    fig.tight_layout()
    return fig, ax


def plot_2dvectors(
    embedding: np.ndarray,
    cluster_labels: Optional[Union[np.ndarray, List]] = None,
    cluster_ids: Optional[List] = None,
    point_ids: Optional[List] = None,
    labels: Optional[List[Union[int, str]]] = None,
    cluster_embeddings: Optional[np.ndarray] = None,
    plot_cluster_embeddings: bool = False,
    show_labels: bool = False,
    label_fontsize: int = 8,
    cluster_label_fontsize: int = 10,
    label_alpha: float = 0.7,
    title: str = "2D Embedding – Points by Cluster",
    centroid_name = 'Feature',
    s: int = 30,
    alpha: float = 0.6,
    x_label: str = "X1",
    y_label: str = "X2",
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D vectors with optional cluster-wise coloring, markers, and legend.

    This function supports:
    - Optional cluster assignments (`cluster_labels`), which can be integers or strings.
    - Automatic `cluster_ids` detection if not provided.
    - Optional cluster anchor points (`cluster_embeddings`).
    - Optional text labels for points and clusters.
    - Composable plotting via `fig_ax`.

    Args:
        embedding: Array of shape (n_samples, 2) containing 2D points.
        cluster_labels: Optional array/list assigning each point to a cluster.
        cluster_ids: Optional list of unique cluster identifiers; defines color and marker order.
        point_ids: Optional IDs aligned with `embedding`.
        labels: Optional custom labels to display instead of `point_ids`.
        cluster_embeddings: Optional array of shape (n_clusters, 2) for cluster anchor points.
        plot_cluster_embeddings: Whether to plot cluster embeddings.
        show_labels: Whether to render text labels for points.
        label_fontsize: Font size for point labels.
        cluster_label_fontsize: Font size for cluster anchor labels.
        label_alpha: Transparency for text labels.
        title: Plot title.
        s: Marker size for points.
        alpha: Transparency of points.
        x_label: Label for X-axis.
        y_label: Label for Y-axis.
        fig_ax: Optional (fig, ax) tuple for composable plotting.

    Returns:
        Tuple (fig, ax) containing the matplotlib figure and axes.
    """

    embedding = np.asarray(embedding)

    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("embedding must have shape (n_samples, 2)")

    if cluster_labels is None:
        cluster_labels = np.zeros(len(embedding))
        if cluster_ids is None:
            cluster_ids = [0]
    else:
        cluster_labels = np.asarray(cluster_labels)
        if cluster_ids is None:
            cluster_ids = sorted(np.unique(cluster_labels))

    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=(10, 8))

    palette = sns.color_palette("tab10", len(cluster_ids))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    
    for i, cluster_id in enumerate(cluster_ids):
        idx = cluster_labels == cluster_id
        if not np.any(idx):
            continue

        plot_labels = None
        if show_labels:
            if labels is not None:
                plot_labels = np.asarray(labels)[idx]
            elif point_ids is not None:
                plot_labels = np.asarray(point_ids)[idx]

        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            marker=markers[i % len(markers)],
            s=s,
            alpha=alpha,
            c=[palette[i % len(palette)]],
            zorder=5
        )

        if show_labels and plot_labels is not None:
            texts = []
            x_range = np.ptp(ax.get_xlim())
            y_range = np.ptp(ax.get_ylim())
            dx = 0.01 * x_range
            dy = 0.01 * y_range

            for (x, y), label in zip(embedding[idx], plot_labels):
                texts.append(
                    ax.text(
                        x + dx,
                        y + dy,
                        str(label),
                        fontsize=label_fontsize,
                        alpha=label_alpha,
                        zorder=10
                    )
                )

            if _HAS_ADJUSTTEXT:
                adjust_text(
                    texts,
                    ax=ax,
                    expand_points=(1.5, 1.5),
                    force_text=0.5,
                    force_points=0.2,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
                )

        if len(cluster_ids) > 1:
            ax.scatter(
                [], [],
                marker=markers[i % len(markers)],
                color=palette[i % len(palette)],
                label=str(cluster_id)
            )

    if plot_cluster_embeddings and cluster_embeddings is not None:
        ax.scatter(
            cluster_embeddings[:, 0],
            cluster_embeddings[:, 1],
            marker="X",
            s=200,
            c="black",
            edgecolors="white",
            linewidths=1.5,
            label=centroid_name,
            zorder=20
        )

        texts = []
        for (x, y), cid in zip(cluster_embeddings, cluster_ids):
            texts.append(
                ax.text(
                    x,
                    y,
                    str(cid),
                    fontsize=cluster_label_fontsize,
                    fontweight="bold",
                    zorder=25
                )
            )

        if _HAS_ADJUSTTEXT:
            adjust_text(
                texts,
                ax=ax,
                expand_points=(1.5, 1.5),
                force_text=0.5,
                force_points=0.5,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)

    if len(cluster_ids) > 1:
        ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_umap_by_feature(
    embedding: np.ndarray,
    feature_assignments: np.ndarray,
    feature_ids: List,
    movie_ids: Optional[List] = None,
    labels: Optional[List[Union[int, str]]] = None,
    feature_embeddings: Optional[np.ndarray] = None,
    plot_feature_embeddings: bool = False,
    show_labels: bool = False,
    label_fontsize: int = 8,
    feature_label_fontsize: int = 10,
    label_alpha: float = 0.7,
    title: str = "UMAP – Movies by Feature",
    s: int = 30,
    alpha: float = 0.6,
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
):
    """
    Plot a 2D UMAP embedding with feature-wise markers and optional labels.
    Labels are automatically adjusted using adjustText to avoid overlapping markers.

    Args:
        embedding: Array of shape (n_samples, 2) containing the 2D embedding of movies.
        feature_assignments: Array assigning each sample to a feature ID.
        feature_ids: List of feature identifiers.
        movie_ids: Optional list of movie IDs aligned with embedding.
        labels: Optional custom labels to display instead of movie_ids.
        feature_embeddings: Optional array of shape (n_features, 2) for feature anchors.
        plot_feature_embeddings: Whether to plot feature embeddings.
        show_labels: Whether to render text labels for each movie.
        label_fontsize: Font size for movie labels.
        feature_label_fontsize: Font size for feature labels.
        label_alpha: Transparency level for text labels.
        title: Plot title.
        s: Marker size for movie points.
        alpha: Transparency of movie points.
        fig_ax: Optional tuple of (figure, axes) for composable plotting.

    Returns:
        (fig, ax)
    """
    
    fig, ax = fig_ax if fig_ax is not None else plt.subplots(figsize=(10, 8))

    palette = sns.color_palette("tab10", len(feature_ids))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']

    for i, feature_id in enumerate(feature_ids):
        idx = feature_assignments == feature_id
        if not np.any(idx):
            continue

        plot_labels = None
        if show_labels:
            if labels is not None:
                plot_labels = np.array(labels)[idx]
            elif movie_ids is not None:
                plot_labels = np.array(movie_ids)[idx]

        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            marker=markers[i % len(markers)],
            s=s,
            alpha=alpha,
            c=[palette[i % len(palette)]],
            zorder=5
        )

        if show_labels and plot_labels is not None:
            texts = []
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            dx = 0.01 * x_range
            dy = 0.01 * y_range

            for (x, y), label in zip(embedding[idx], plot_labels):
                texts.append(
                    ax.text(
                        x + dx, y + dy, str(label),
                        fontsize=label_fontsize,
                        alpha=label_alpha,
                        zorder=10 
                    )
                )

            adjust_text(
                texts,
                ax=ax,
                expand_points=(1.5, 1.5),
                force_text=0.5,
                force_points=0.2,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )

        ax.scatter(
            [], [],
            marker=markers[i % len(markers)],
            color=palette[i % len(palette)],
            label=f"{feature_id}"
        )

    if plot_feature_embeddings and feature_embeddings is not None:
        ax.scatter(
            feature_embeddings[:, 0],
            feature_embeddings[:, 1],
            marker="X",
            s=200,
            c="black",
            edgecolors="white",
            linewidths=1.5,
            label="Features",
            zorder=20  
        )

        texts = []
        for (x, y), fid in zip(feature_embeddings, feature_ids):
            texts.append(
                ax.text(
                    x, y,
                    str(fid),
                    fontsize=feature_label_fontsize,
                    fontweight="bold",
                    zorder=25
                )
            )

        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.5, 1.5),
            force_text=0.5,
            force_points=0.5,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
        )

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax

