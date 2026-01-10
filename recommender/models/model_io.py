import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import re
from ..utils.helpers import load_objects
from ..plots.plotting import (
    save_plot,
    plot_metric_scatter,
    plot_als_curves_multiple_models
)


def add_suffix_model_info(
    filename: Union[str, Path],
    fitted_model: dict,
) -> Path:
    """
    Append fitted model metadata to a filename.

    Args:
        filename:
            Input file path.
        fitted_model:
            Dictionary-like object containing model parameters.
            Required keys:
                - K (int)
                - nepochs (int)
                - precision (float)
                - vector_penality (float)
                - mode (str)

    Returns:
            New path with model information appended to the filename.
    """
    K = fitted_model.get('K')
    nepochs = fitted_model.get('nepochs')
    max_iter = fitted_model.get('max_iter')
    precision = fitted_model.get('precision')
    vector_penality = fitted_model.get('vector_penality')
    mode = fitted_model.get('mode')
    if nepochs is not None:
        if nepochs < max_iter:
            epochs_str =  f"-best_nepochs{nepochs}"
        else:
            epochs_str = ''
    else:
        epochs_str = ''

    path = Path(filename)

    suffix = (
        f"-K{K:02d}"
        f"{epochs_str}"
        f"-maxiter{max_iter}"
        f"-lambda{precision}"
        f"-tau{vector_penality}"
        f"-{mode}"
    )

    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def load_models(models_path: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load trained model(s) from disk.
    
    Args:
        models_path: Path to a model file or a list of model paths.
    
    Returns:
        Dictionary containing model parameters and metadata,
        or a list of such dictionaries if multiple paths are provided.
    """

    def _load_model(model_path: str) -> Dict[str, Any]:
        fitted_model = load_objects(model_path)
    
        return fitted_model
        


    if isinstance(models_path, list):
        return [_load_model(path) for path in models_path]

    return _load_model(models_path)


def get_models_path_filtered(
    model_folder: str,
    K_list: Optional[List[int]] = None,
    vector_penality_list: Optional[List[float]] = None,
    max_iter_list:Optional[List[int]] = None,
    mode: str = "",
    pattern: str = "",
    exclude_pattern: Optional[str] = None,
    vector_penality_tol: float = 1e-8
) -> List[str]:
    """
    Get all model paths from folder (filtered).
    
    Args:
        model_folder: Directory containing model files.
        K_list: List of embedding dimensions to filter by.
        vector_penality_list: List of vector penality values to filter by.
        max_iter_list: List of  max iteration
        mode: "bias_only" or "with_embeddings" or "with_features" to filter by model type.
        pattern: Additional regex pattern to include (applied to path).
        exclude_pattern: Regex pattern to exclude (applied to path).
        vector_penality_tol: Absolute tolerance for float comparison.
    
    Returns:
        List of model paths.
    """

    if K_list is None:
        K_list = []

    if vector_penality_list is None:
        vector_penality_list = []

    models_path = []

    for path in Path(model_folder).rglob("*.joblib"):
        path_str = str(path)

        if pattern and not re.search(pattern, path_str):
            continue

        if exclude_pattern and re.search(exclude_pattern, path_str):
            continue

        try:
            results = load_objects(path_str)
        except Exception:
            continue

        if K_list and results.get("K") not in K_list:
            continue
        if vector_penality_list:
            vp = results.get("vector_penality")
            if vp is None:
                continue
            if not np.any([abs(vp - v) <= vector_penality_tol for v in vector_penality_list]):
                continue
        if max_iter_list:
            mi = results.get("max_iter")
            if mi is None:
                continue
            if mi not in max_iter_list:
                continue
        if mode and results.get("mode") != mode:
            continue

        models_path.append(path_str)

    return models_path


def load_models_filtered(
    model_folder: str,
    K_list: Optional[List[int]] = None,
    vector_penality_list: Optional[List[float]] = None,
    max_iter_list:Optional[List[int]]=None,
    mode: str = "",
    pattern: str = "",
    exclude_pattern: Optional[str] = None,
    vector_penality_tol: float = 1e-8
)-> List[str]:
    """
    Load all models value from folder (filtered).
    
    Args:
        model_folder: Directory containing model files.
        K_list: List of embedding dimensions to filter by.
        vector_penality_list: List of vector penality values to filter by.
        max_iter_list: List of max iterations.
        mode: "bias_only" or "with_embeddings" or "with_features" to filter by model type.
        pattern: Additional regex pattern to include (applied to path).
        exclude_pattern: Regex pattern to exclude (applied to path).
        vector_penality_tol: Absolute tolerance for float comparison.
    
    Returns:
        List of loaded model dictionaries.
    """
    models_path = get_models_path_filtered(
        model_folder,
        K_list,
        vector_penality_list,
        max_iter_list,
        mode,
        pattern,
        exclude_pattern,
        vector_penality_tol
    )
    models = []
    for path in models_path:
        models.append(load_models(path))

    return models


def get_models_metrics(
    models_path: List[Path]
) -> tuple:
    """
    Extract metrics from multiple model files.
    
    Args:
        models_path: List of paths to model files.
    
    Returns:
        Tuple of metric lists (train_loss, test_loss, train_rmse, test_rmse,
                            precision, vector_penality, bias_penality, K_list,
                            iterations, mode_list).
    """
    train_last_loss_list = []
    test_last_loss_list = []
    train_last_rmse_list = []
    test_last_rmse_list = []
    precision_list = []
    vector_penality_list = []
    bias_penality_list = []
    K_list = []
    iteration_list = []
    mode_list = []

    for path in models_path:
        model = load_models(path)
        user_vectors = model.get("user_vectors")
        bias_only = (user_vectors is None) or (len(user_vectors) == 0)

        feature_vectors = model.get("feature_vectors")
        with_features = (feature_vectors is not None) and (len(feature_vectors) > 0)
        if bias_only:
            model["mode"] = "bias_only"
        elif with_features:
            model["mode"] = "with_features"
        else:
            model["mode"] = "with_embeddings"
        
        train_last_loss_list.append(model["train_loss_hist"][-1])
        test_last_loss_list.append(model["test_loss_hist"][-1])
        train_last_rmse_list.append(model["train_rmse_hist"][-1])
        test_last_rmse_list.append(model["test_rmse_hist"][-1])
        precision_list.append(model["precision"])
        vector_penality_list.append(model["vector_penality"])
        bias_penality_list.append(model["bias_penality"])
        K_list.append(model["K"])
        iteration_list.append(len(model["test_rmse_hist"]))
        mode_list.append(model["mode"])
        
    return (
        train_last_loss_list,
        test_last_loss_list,
        train_last_rmse_list,
        test_last_rmse_list,
        precision_list,
        vector_penality_list,
        bias_penality_list,
        K_list,
        iteration_list,
        mode_list,
    )


def save_model_metrics(
    fitted_model: Dict,
    toplot: str,
    plot_folder: Path = None,
    save: bool = False,
    overwrite: bool = False
) -> str:
    """
    Save model training metrics plot.
    
    Args:
        fitted_model: Model dictionary.
        toplot: Which metric to plot ('loss', 'rmse', or 'both').
        plot_folder: Directory to save plot.
        save: Whether to save the plot.
        overwrite: Whether to overwrite existing files.
    
    Returns:
        Path where plot was saved.
    """
    if not plot_folder:
        plot_folder = Path(".") / "plots" / "models"
    
    filename = f"{toplot}-model.pdf"
    filename = add_suffix_model_info(filename, fitted_model)
    plot_filepath = plot_folder / filename
    plot_filepath = str(plot_filepath)

    fig, ax = plt.subplots(1, figsize=(6, 4))

    plot_als_curves_multiple_models(
        [fitted_model],
        fig_ax=(fig, ax),
        toplot=toplot,
        with_suptitle=True)
    ax.set_title(None)

    if save:
        save_plot(
            plot_filepath,
            fig=None,
            formats="pdf",
            timestamps=False,
            use_tex=False,
            dpi=300,
            overwrite=overwrite
        )
        print(f"Plot saved to {plot_filepath}")
        
    return plot_filepath


def get_best_iteration_rmse_dataframe(
    nepochs_list: List[int],
    vector_penality_list: List[float],
    K_list: List[int],
    rmse_list: List[float],
    table_filepath: Optional[str]='best_iteration.tex',
    save: bool = True,
) -> pd.DataFrame:
    """
    Build a pivoted metrics DataFrame indexed by tau and columns K,
    with LaTeX-formatted values: rmse [nepochs].

    Args:
        nepochs_list: Number of epochs used for each model.
        vector_penality_list: Vector regularization values (tau).
        K_list: Latent dimension values.
        rmse_list: RMSE for each model.
        table_filepath: Path where the LaTeX table is saved.
        save: Whether to save the DataFrame as a LaTeX table.

    Returns:
        Pivoted pandas DataFrame.
    """
    records = []
    table_filepath = Path(table_filepath)
    table_filepath.parent.mkdir(exist_ok=True, parents=True)
    for nepochs, tau, K, rmse in zip(
        nepochs_list,
        vector_penality_list,
        K_list,
        rmse_list,
    ):
        records.append(
            {
                "tau": tau,
                "K": K,
                "value": f"{nepochs} [{rmse:.4f}]",
            }
        )

    df = (
        pd.DataFrame(records)
        .pivot(index="tau", columns="K", values="value")
        .sort_index(axis=1)
    )

    df.index = df.index.map(lambda x: f"{x:.2g}")

    if save:
        df.to_latex(table_filepath, escape=False)
        print(f"saved in {table_filepath}")

    return df


def save_metrics_grid(
    K_list: List[int],
    vector_penality_list: List[float],
    test_last_rmse_list: List[float],
    plot_filepath: Path,
    title: str = "",
    save: bool = True,
    overwrite: bool = False
) -> Path:
    """
    Save grid plot of metrics vs hyperparameters.
    
    Args:
        K_list: List of K values.
        vector_penality_list: List of tau values.
        test_last_rmse_list: List of test RMSE values.
        plot_filepath: Path to save plot.
        title: Plot title.
        save: Whether to save the plot.
        overwrite: Whether to overwrite existing files.
    
    Returns:
        Path where plot was saved.
    """
    
    plot_filepath = Path(plot_filepath)
    plot_filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_metric_scatter(
        K_list,
        vector_penality_list,
        test_last_rmse_list,
        title=title,
        fig_ax=None
    )

    if save:
        save_plot(
            str(plot_filepath),
            fig=fig,
            formats="pdf",
            timestamps=False,
            use_tex=False,
            dpi=300,
            overwrite=overwrite
        )
        print(f"Plot saved to {plot_filepath}")

    return plot_filepath