import numpy as np
from  pathlib import Path
from typing import Tuple, Callable, List, Dict, Any, Optional
from numba import njit, prange
from tqdm.auto import tqdm
from datetime import datetime

from ..data.csr import (
    get_row_from_csr,
    get_nrows_csr,
    remove_element,
    get_nelements_by_row_csr
)
from ..utils.helpers import (
    show_execution_duration, dump_objects,
    argsort_array,
    sort_array
    )
from ..data.stats import (
    get_upper_outlier,
    get_lower_outlier
)
from ..plots.plotting import plot_als_curves
from ..utils.logging import log_als_run_to_yaml

from .model_io import add_suffix_model_info

@njit(fastmath=True)
def compute_metrics_numba_csr_bias_only(
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    bias_penality: float
) -> Tuple[float, float]:
    """
    Calculate RMSE and loss for bias-only model.
    
    Args:
        user_biases: User bias vector.
        item_biases: Item bias vector.
        ratings_by_user_csr: Ratings CSR matrix.
        precision: Precision parameter (lambda).
        bias_penality: Bias regularization parameter.
    
    Returns:
        Tuple of (rmse, loss).
    """
    M = get_nrows_csr(ratings_by_user_csr)

    se = 0.0
    cnt = 0

    for m in range(M):
        items, ratings = get_row_from_csr(ratings_by_user_csr, m)
        nitems = len(items)
        if nitems == 0:
            continue

        pred = item_biases[items] + user_biases[m]
        err = ratings - pred

        se += np.sum(err ** 2)
        cnt += nitems

    rmse = np.sqrt(se / cnt)
    loss = (
        precision / 2.0 * se
        + bias_penality / 2.0 * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
    )
    return rmse, loss


@njit(parallel=True, fastmath=True)
def als_numba_iteration_numba_csr_bias_only(
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_item_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    bias_penality: float,
    K: int
) -> None:
    """
    Single ALS iteration for bias-only model.
    
    Args:
        user_biases: User bias vector (modified in-place).
        item_biases: Item bias vector (modified in-place).
        ratings_by_user_csr: Ratings by user CSR matrix.
        ratings_by_item_csr: Ratings by item CSR matrix.
        precision: Precision parameter.
        bias_penality: Bias regularization parameter.
        K: Embedding dimension (unused but kept for consistency).
    """
    M = get_nrows_csr(ratings_by_user_csr)
    N = get_nrows_csr(ratings_by_item_csr)

    def update_user_bias_only(m: int):
        items, ratings = get_row_from_csr(ratings_by_user_csr, m)
        nitems = len(items)
        if nitems == 0:
            return user_biases[m]
        item_bias = item_biases[items]
        pred = item_bias
        resid = ratings - pred

        s = precision * np.sum(resid)
        user_bias = s / (precision * nitems + bias_penality)
        return user_bias

    def update_item_bias_only(n: int):
        users, ratings = get_row_from_csr(ratings_by_item_csr, n)
        nusers = len(users)
        if nusers == 0:
            return item_biases[n]

        user_bias = user_biases[users]
        pred = user_bias
        resid = ratings - pred

        s = precision * np.sum(resid)
        item_bias = s / (precision * nusers + bias_penality)
        return item_bias

    for m in prange(M):
        user_biases[m] = update_user_bias_only(m)

    for n in prange(N):
        item_biases[n] = update_item_bias_only(n)


@show_execution_duration(return_time=True)
def als_update_biases_numba_csr_bias_only(
    ratings_by_user_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_user_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    bias_penality: float,
    K: int,
    max_iter: int = 10
) -> Tuple:
    """
    Train bias-only ALS model.
    
    Args:
        ratings_by_user_train_csr: Training ratings by user.
        ratings_by_movie_train_csr: Training ratings by movie.
        ratings_by_user_test_csr: Test ratings by user.
        ratings_by_movie_test_csr: Test ratings by movie.
        precision: Precision parameter.
        bias_penality: Bias regularization parameter.
        K: Embedding dimension (unused but kept for consistency).
        max_iter: Maximum number of iterations.
    
    Returns:
        Tuple of (user_biases, item_biases, train_loss_hist, train_rmse_hist,
                test_loss_hist, test_rmse_hist).
    """
    M = get_nrows_csr(ratings_by_user_train_csr)
    N = get_nrows_csr(ratings_by_movie_train_csr)

    user_biases = np.zeros(M)
    item_biases = np.zeros(N)

    train_rmse_hist = []
    train_loss_hist = []
    test_rmse_hist = []
    test_loss_hist = []

    for it in tqdm(range(max_iter), unit="epoch"):
        als_numba_iteration_numba_csr_bias_only(
            user_biases, item_biases,
            ratings_by_user_train_csr, ratings_by_movie_train_csr,
            precision, bias_penality, K)

        train_rmse, train_loss = compute_metrics_numba_csr_bias_only(
            user_biases, item_biases,
            ratings_by_user_train_csr,
            precision, bias_penality)
        test_rmse, test_loss = compute_metrics_numba_csr_bias_only(
            user_biases, item_biases,
            ratings_by_user_test_csr,
            precision, bias_penality)

        train_rmse_hist.append(train_rmse)
        train_loss_hist.append(train_loss)
        test_rmse_hist.append(test_rmse)
        test_loss_hist.append(test_loss)

        if (it + 1) % 10 == 0 or it == 0:
            print("")
            print(f"epoch {it+1}/{max_iter} | train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} | train_rmse={train_rmse:.4f} "
                f"test_rmse={test_rmse:.4f}")

    return (user_biases, item_biases,
            train_loss_hist, train_rmse_hist, test_loss_hist, test_rmse_hist)


def compute_metrics_csr(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float
) -> Tuple[float, float]:
    """
    Calculate RMSE and loss for full ALS model.
    
    Args:
        user_vectors: User embedding matrix.
        item_vectors: Item embedding matrix.
        user_biases: User bias vector.
        item_biases: Item bias vector.
        ratings_by_user_csr: Ratings CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
    
    Returns:
        Tuple of (rmse, loss).
    """
    M = get_nrows_csr(ratings_by_user_csr)

    se = 0.0
    cnt = 0

    for m in range(M):
        items, ratings = get_row_from_csr(ratings_by_user_csr, m)
        nitems = len(items)
        if nitems == 0:
            continue

        V = item_vectors[items]
        u = user_vectors[m]
        pred = V @ u + item_biases[items] + user_biases[m]
        err = ratings - pred

        se += np.sum(err ** 2)
        cnt += nitems

    rmse = np.sqrt(se / cnt)
    loss = (
        precision / 2.0 * se
        + bias_penality / 2.0 * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
        + vector_penality / 2.0 * (np.sum(user_vectors ** 2) + np.sum(item_vectors ** 2))
    )
    return rmse, loss


@njit(parallel=False, fastmath=True)
def compute_metrics_numba_csr(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float
) -> Tuple[float, float]:
    """
    Calculate RMSE and loss for full ALS model.
    
    Args:
        user_vectors: User embedding matrix.
        item_vectors: Item embedding matrix.
        user_biases: User bias vector.
        item_biases: Item bias vector.
        ratings_by_user_csr: Ratings CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
    
    Returns:
        Tuple of (rmse, loss).
    """
    M = get_nrows_csr(ratings_by_user_csr)

    se_by_user = np.zeros(M, dtype=float)
    nratings_by_user = np.zeros(M, dtype=float)

    for m in range(M):
        items, ratings = get_row_from_csr(ratings_by_user_csr, m)
        nitems = len(items)
        nratings_by_user[m] = nitems
        if nitems == 0:
            continue

        V = item_vectors[items]
        u = user_vectors[m]
        pred = V @ u + item_biases[items] + user_biases[m]
        err = ratings - pred
        se_by_user[m] = np.sum(err ** 2)
    cnt = np.sum(nratings_by_user)
    se = np.sum(se_by_user)
    rmse = np.sqrt(se / cnt)
    loss = (
        precision / 2.0 * se
        + bias_penality / 2.0 * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
        + vector_penality / 2.0 * (np.sum(user_vectors ** 2) + np.sum(item_vectors ** 2))
    )
    return rmse, loss
    
@njit(parallel=True, fastmath=True)
def compute_rmse_by_user_numba_csr(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Calculate RMSE and loss for full ALS model.
    
    Args:
        user_vectors: User embedding matrix.
        item_vectors: Item embedding matrix.
        user_biases: User bias vector.
        item_biases: Item bias vector.
        ratings_by_user_csr: Ratings CSR matrix.
    
    Returns:
        RMSE for each item.
    """
    M = get_nrows_csr(ratings_by_user_csr)

    rmse_by_users = np.full(M, np.inf, dtype=float)
    
    for m in prange(M):
        items, ratings = get_row_from_csr(ratings_by_user_csr, m)
        nitems = len(items)
        if nitems == 0:
            continue

        u = user_vectors[m]
        V = item_vectors[items]
        pred = V @ u + item_biases[items] + user_biases[m]
        err = ratings - pred
        
        rmse_by_users[m] = np.sqrt(np.mean(err ** 2))

    return rmse_by_users


@njit(parallel=True, fastmath=True)
def compute_rmse_by_item_numba_csr(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Calculate RMSE and loss for full ALS model.
    
    Args:
        user_vectors: User embedding matrix.
        item_vectors: Item embedding matrix.
        user_biases: User bias vector.
        item_biases: Item bias vector.
        ratings_by_movie_csr: Ratings CSR matrix.
    
    Returns:
        RMSE for each item.
    """
    N = get_nrows_csr(ratings_by_movie_csr)

    rmse_by_items = np.zeros(N, dtype=float)
    
    for n in prange(N):
        users, ratings = get_row_from_csr(ratings_by_movie_csr, n)
        nusers = len(users)
        if nusers == 0:
            continue

        U = user_vectors[users]
        v = item_vectors[n]
        pred = U @ v + item_biases[n] + user_biases[users]
        err = ratings - pred
        
        rmse_by_items[n] = np.sqrt(np.mean(err ** 2))

    return rmse_by_items


def early_stop(
    state: Dict,
    score: float,
    patience: int = 10,
    min_delta: float = 1e-4,
    higher_is_better: bool = False
) -> bool:
    """
    Online early stopping callback.

    Args:
        state: Mutable dict tracking the stopping state.
            Keys used:
                - best: best score so far
                - best_epoch: epoch of best score
                - no_improve: consecutive epochs without improvement
                - epoch: current epoch count
        score: Current validation metric
        patience: Number of non-improving epochs allowed
        min_delta: Minimum improvement to reset patience
        higher_is_better: True if higher score is better, False if lower is better

    Returns:
        True if training should stop, False otherwise
    """
    state.setdefault("best", float("-inf") if higher_is_better else float("inf"))
    state.setdefault("best_epoch", None)
    state.setdefault("no_improve", 0)
    state.setdefault("epoch", -1)
    state["epoch"] += 1
    
    improved = False
    if higher_is_better:
        if score - state["best"] > min_delta:
            improved = True
    else:
        if state["best"] - score > min_delta:
            improved = True

    if improved:
        state["best"] = score
        state["best_epoch"] = state["epoch"]
        state["no_improve"] = 0
    else:
        state["no_improve"] += 1
        
    return state["no_improve"] >= patience


def offline_best_epoch_from_scores(
    scores: List[float],
    patience: int = 10,
    min_delta: float = 1e-4,
    higher_is_better: bool = False
) -> Tuple[int, Optional[int], Dict]:
    """
    Offline early stopping using the online early_stop callback.

    Args:
        scores: Validation scores per epoch.
        patience: Number of non-improving epochs allowed.
        min_delta: Minimum improvement to reset patience.
        higher_is_better: True if higher score is better.

    Returns:
        Dict with keys:
            - best: best score observed
            - best_epoch: epoch index of best score
            - no_improve: number of consecutive non-improving epochs
    """
    state: Dict = {}
    for score in scores:
        should_stop = early_stop(
            state=state,
            score=score,
            patience=patience,
            min_delta=min_delta,
            higher_is_better=higher_is_better,
        )
        if should_stop:
            break

    return {k: state[k] for  k in ['best', 'best_epoch', 'no_improve']}



def get_underfitting(
    train_rmses: np.ndarray,
    test_rmses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect underfitting users based on RMSE outliers.

    A user is considered underfitting if they are an upper IQR
    outlier in both train and test RMSE distributions.

    Args:
        train_rmses: Training RMSE per user (shape: [n_users]).
        test_rmses: Test RMSE per user (shape: [n_users]).

    Returns:
        user_indices:
            Indices of users classified as underfitting,
            sorted by descending test RMSE.
        rmse_gaps:
            RMSE gap per user defined as
            (train_rmses - test_rmses),
            ordered to match `user_indices`.
    """
    _, train_idx = get_upper_outlier(train_rmses)
    _, test_idx = get_upper_outlier(test_rmses)

    common_idx = np.intersect1d(train_idx, test_idx)

    rmse_gap = train_rmses[common_idx] - test_rmses[common_idx]

    order = argsort_array(
        test_rmses[common_idx],
        descending=True
    )

    return common_idx[order], rmse_gap[order]


def get_overfitting(
    train_rmses: np.ndarray,
    test_rmses: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect overfitting users based on RMSE gaps.

    Overfitting users are identified as lower IQR outliers
    of the RMSE gap (train_rmses - test_rmses), i.e. users
    with unusually worse test performance compared to train.

    Args:
        train_rmses: Training RMSEs.
        test_rmses: Test RMSEs.

    Returns:
        user_indices:
            Indices of users classified as overfitting,
            sorted by most negative RMSE gap first.
        rmse_gaps:
            RMSE gaps (train_rmses - test_rmses),
            ordered to match `user_indices`.
    """
    rmse_gap = train_rmses - test_rmses

    gaps, indices = get_lower_outlier(rmse_gap)

    return indices, gaps

@njit(parallel=True, fastmath=True)
def als_update_users_numba_csr(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float,
    K: int
) -> None:
    """
    Single ALS iteration.
    
    Args:
        user_vectors: User embedding matrix (modified in-place).
        item_vectors: Item embedding matrix (modified in-place).
        user_biases: User bias vector (modified in-place).
        item_biases: Item bias vector (modified in-place).
        ratings_by_user_csr: Ratings by user CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        K: Embedding dimension.
    """
    M = get_nrows_csr(ratings_by_user_csr)

    def update_user(m: int):
        items, ratings = get_row_from_csr(ratings_by_user_csr, m)
        nitems = len(items)
        if nitems == 0:
            return user_biases[m], user_vectors[m]

        V = item_vectors[items]
        u = user_vectors[m]
        item_bias = item_biases[items]
        pred = V @ u.T + item_bias
        resid = ratings - pred

        s = precision * np.sum(resid)
        user_bias = s / (precision * nitems + bias_penality)

        resid2 = ratings - user_bias - item_bias

        linsum_v = precision * (V.T @ resid2.T)
        transition_matrix = vector_penality * np.identity(K) + precision * (V.T @ V)

        user_vector = np.linalg.solve(transition_matrix, linsum_v)
        return user_bias, user_vector

    for m in prange(M):
        user_biases[m], user_vectors[m, :] = update_user(m)

@njit(parallel=True, fastmath=True)
def als_update_items_numba_csr(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_item_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float,
    K: int
) -> None:
    """
    Single ALS iteration.
    
    Args:
        user_vectors: User embedding matrix (modified in-place).
        item_vectors: Item embedding matrix (modified in-place).
        user_biases: User bias vector (modified in-place).
        item_biases: Item bias vector (modified in-place).
        ratings_by_item_csr: Ratings by item CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        K: Embedding dimension.
    """
    N = get_nrows_csr(ratings_by_item_csr)

    def update_item(n: int):
        users, ratings = get_row_from_csr(ratings_by_item_csr, n)
        nusers = len(users)
        if nusers == 0:
            return item_biases[n], item_vectors[n]

        U = user_vectors[users]
        v = item_vectors[n]
        user_bias = user_biases[users]
        pred = U @ v.T + user_bias
        resid = ratings - pred

        s = precision * np.sum(resid)
        item_bias = s / (precision * nusers + bias_penality)

        resid2 = ratings - user_bias - item_bias

        linsum_u = precision * (U.T @ resid2.T)
        transition_matrix = vector_penality * np.identity(K) + precision * (U.T @ U)

        item_vector = np.linalg.solve(transition_matrix, linsum_u)
        return item_bias, item_vector

    for n in prange(N):
        item_biases[n], item_vectors[n, :] = update_item(n)
        
@njit
def als_iteration_numba_csr(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_item_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float,
    K: int
) -> None:
    """
    Single ALS iteration.
    
    Args:
        user_vectors: User embedding matrix (modified in-place).
        item_vectors: Item embedding matrix (modified in-place).
        user_biases: User bias vector (modified in-place).
        item_biases: Item bias vector (modified in-place).
        ratings_by_user_csr: Ratings by user CSR matrix.
        ratings_by_item_csr: Ratings by item CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        K: Embedding dimension.
    """
    als_update_users_numba_csr(
        user_vectors,
        item_vectors,
        user_biases,
        item_biases,
        ratings_by_user_csr,
        precision,
        vector_penality,
        bias_penality,
        K
    )

    als_update_items_numba_csr(
        user_vectors,
        item_vectors,
        user_biases,
        item_biases,
        ratings_by_item_csr,
        precision,
        vector_penality,
        bias_penality,
        K
    )


@show_execution_duration(return_time=True)
def als_update_embeddings_and_biases_numba_csr(
    ratings_by_user_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_user_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    bias_penality: float,
    vector_penality: float,
    K: int,
    max_iter: int = 10,
    is_early_stopping = True,
    patience = 5,
    min_delta = 1e-3,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    Train full ALS model with embeddings and biases.
    
    Args:
        ratings_by_user_train_csr: Training ratings by user.
        ratings_by_movie_train_csr: Training ratings by movie.
        ratings_by_user_test_csr: Test ratings by user.
        ratings_by_movie_test_csr: Test ratings by movie.
        precision: Precision parameter.
        bias_penality: Bias regularization parameter.
        vector_penality: Vector regularization parameter.
        K: Embedding dimension.
        max_iter: Maximum number of iterations.
        is_early_stopping:,
        patience:,
        min_delta:,
        seed: Random seed.
    Returns:
        ...
    """
    M = get_nrows_csr(ratings_by_user_train_csr)
    N = get_nrows_csr(ratings_by_movie_train_csr)

    np.random.seed(seed)
    mean = np.zeros(K)
    cov = (1.0 / np.sqrt(K)) * np.identity(K)
    user_vectors = np.random.multivariate_normal(mean, cov, M)
    item_vectors = np.random.multivariate_normal(mean, cov, N)
    user_biases = np.zeros(M)
    item_biases = np.zeros(N)

    state = {}
    mode = "with_embeddings"        
    bias_only = False
    fitted_model = {
        "mode": mode,
        "user_vectors": user_vectors,
        "item_vectors": item_vectors,
        "feature_vectors": [],
        "user_biases": user_biases,
        "item_biases": item_biases,
        "precision": precision,
        "bias_penality": bias_penality,
        "vector_penality": vector_penality,
        "K": K,
        "bias_only": bias_only,
        "train_loss_hist": [],
        "train_rmse_hist": [],
        "test_loss_hist": [],
        "test_rmse_hist": [],
        "max_iter": max_iter,
        "nepochs": 0, 
        "best_epoch": state.get('best_epoch'),
        "no_improve": state.get('no_improve')
    }
    fitted_model_best = {
        **fitted_model,
        "train_loss_hist": [],
        "train_rmse_hist": [],
        "test_loss_hist": [],
        "test_rmse_hist": [],
    }

    for it in tqdm(range(max_iter), unit="epoch"):
        als_iteration_numba_csr(
            user_vectors, item_vectors, user_biases, item_biases,
            ratings_by_user_train_csr, ratings_by_movie_train_csr,
            precision, vector_penality, bias_penality, K
            )

        train_rmse, train_loss = compute_metrics_numba_csr(
            user_vectors, item_vectors,
            user_biases, item_biases,
            ratings_by_user_train_csr,
            precision, vector_penality, bias_penality)
        test_rmse, test_loss = compute_metrics_numba_csr(
            user_vectors, item_vectors,
            user_biases, item_biases,
            ratings_by_user_test_csr,
            precision, vector_penality, bias_penality)

        fitted_model['train_rmse_hist'].append(train_rmse)
        fitted_model['train_loss_hist'].append(train_loss)
        fitted_model['test_rmse_hist'].append(test_rmse)
        fitted_model['test_loss_hist'].append(test_loss)

        fitted_model_best['train_rmse_hist'].append(train_rmse)
        fitted_model_best['train_loss_hist'].append(train_loss)
        fitted_model_best['test_rmse_hist'].append(test_rmse)
        fitted_model_best['test_loss_hist'].append(test_loss)
        
        should_stop = early_stop(
            state=state,
            score=test_rmse,
            patience=patience,
            min_delta=min_delta,
        )
        if state.get('best_epoch') == it:
            fitted_model_best['user_vectors'] = np.copy(user_vectors)
            fitted_model_best['item_vectors'] = np.copy(item_vectors)
            fitted_model_best['user_biases'] = np.copy(user_biases)
            fitted_model_best['item_biases'] = np.copy(item_biases)
        
        if (it + 1) % 10 == 0 or it == 0:
            training_info_str = (
                f"epoch {it+1}/{max_iter} | train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} | train_rmse={train_rmse:.4f} "
                f"test_rmse={test_rmse:.4f} |"
                f"best_epoch {state.get('best_epoch')}"
            )
        
            tqdm.write(training_info_str)
            
        if is_early_stopping and should_stop:
            break
    fitted_model.update({            
        "nepochs": it + 1, 
        "best_epoch": state.get('best_epoch'),
    })
    fitted_model_best.update(
        {
        "nepochs": state.get('best_epoch') + 1,
        "best_epoch": state.get('best_epoch'),
        "no_improve": state.get('no_improve') 
        }
    )
    for k in ['train_rmse_hist', 'train_loss_hist', 'test_rmse_hist', 'test_loss_hist']:
        fitted_model_best[k] = fitted_model_best[k][: fitted_model_best['nepochs']]
    return fitted_model_best, fitted_model


@njit(parallel=True, fastmath=True)
def compute_metrics_numba_csr_with_features(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    feature_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    features_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    movies_by_feature_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float,
    with_features: bool = False
) -> Tuple[float, float]:
    """
    Calculate RMSE and loss for feature regularization ALS model.
    
    Args:
        user_vectors: User embedding matrix.
        item_vectors: Item embedding matrix.
        feature_vectors: Feature embedding matrix.
        user_biases: User bias vector.
        item_biases: Item bias vector.
        ratings_by_movie_csr: Ratings by movie CSR matrix.
        features_by_movie_csr: Features by movie CSR matrix.
        movies_by_feature_csr: Movies by feature CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        with_features: Whether to include feature regularization.
    
    Returns:
        Tuple of (rmse, loss).
    """
    
    N = get_nrows_csr(ratings_by_movie_csr)
    K = user_vectors.shape[1]
    vmean_vectors = np.zeros((N, K))
    
    se_by_item = np.zeros(N, dtype=float)
    nratings_by_item = np.zeros(N, dtype=float)
    for n in prange(N):
        users, ratings = get_row_from_csr(ratings_by_movie_csr, n)
        nusers = len(users)
        nratings_by_item[n] = nusers
        if nusers == 0:
            continue
        
        U = user_vectors[users]
        v = item_vectors[n]
        pred = U @ v + item_biases[n] + user_biases[users]
        err = ratings - pred
        se_by_item[n] = np.sum(err ** 2)

        features, _ = get_row_from_csr(features_by_movie_csr, n)
        Fn = len(features)
        vmean_vectors[n, :] = 1 / np.sqrt(Fn) * np.sum(feature_vectors[features, :], axis=0)

        
    cnt = np.sum(nratings_by_item)
    se = np.sum(se_by_item) 
    rmse = np.sqrt(se / cnt)
    loss = (
        precision / 2.0 * se
        + bias_penality / 2.0 * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
        + vector_penality / 2.0 * (
            + np.sum((item_vectors - with_features * vmean_vectors) ** 2)
            + np.sum(user_vectors ** 2)
            + with_features * np.sum(vmean_vectors ** 2)
        )
    )
    return rmse, loss


def compute_metrics_csr_with_features(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    feature_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    features_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    movies_by_feature_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float,
    with_features: bool = False
) -> Tuple[float, float]:
    """
    Calculate RMSE and loss for feature regularization ALS model.
    
    Args:
        user_vectors: User embedding matrix.
        item_vectors: Item embedding matrix.
        feature_vectors: Feature embedding matrix.
        user_biases: User bias vector.
        item_biases: Item bias vector.
        ratings_by_movie_csr: Ratings by movie CSR matrix.
        features_by_movie_csr: Features by movie CSR matrix.
        movies_by_feature_csr: Movies by feature CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        with_features: Whether to include feature regularization.
    
    Returns:
        Tuple of (rmse, loss).
    """
    N = get_nrows_csr(ratings_by_movie_csr)
    K = user_vectors.shape[1]
    se = 0.0
    cnt = 0
    vmean_vectors = np.zeros((N, K))
    
    for n in range(N):
        users, ratings = get_row_from_csr(ratings_by_movie_csr, n)
        nusers = len(users)
        if nusers == 0:
            continue

        U = user_vectors[users]
        v = item_vectors[n]
        pred = U @ v + item_biases[n] + user_biases[users]
        err = ratings - pred

        se += np.sum(err ** 2)
        cnt += nusers

        features, _ = get_row_from_csr(features_by_movie_csr, n)
        Fn = len(features)
        vmean_vectors[n, :] = 1 / np.sqrt(Fn) * np.sum(feature_vectors[features, :], axis=0)

    rmse = np.sqrt(se / cnt)
    loss = (
        precision / 2.0 * se
        + bias_penality / 2.0 * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
        + vector_penality / 2.0 * (
            + np.sum((item_vectors - with_features * vmean_vectors) ** 2)
            + np.sum(user_vectors ** 2)
            + with_features * np.sum(vmean_vectors ** 2)
        )
    )
    return rmse, loss

@njit(parallel=True, fastmath=True)
def als_update_items_numba_csr_with_features(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    feature_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_item_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    features_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float,
    K: int,
    with_features: bool = False
) -> None:
    """
    Single ALS iteration with feature regularization.
    
    Args:
        user_vectors: User embedding matrix (modified in-place).
        item_vectors: Item embedding matrix (modified in-place).
        user_biases: User bias vector (modified in-place).
        item_biases: Item bias vector (modified in-place).
        ratings_by_item_csr: Ratings by item CSR matrix.
        features_by_movie_csr: Features by movie CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        K: Embedding dimension.
        with_features: Whether to use item's features.
    """
    
    N = get_nrows_csr(ratings_by_item_csr)

    def update_item(n: int):
        users, ratings = get_row_from_csr(ratings_by_item_csr, n)
        nusers = len(users)
        if nusers == 0:
            return item_biases[n], item_vectors[n]

        U = user_vectors[users]
        v = item_vectors[n]
        user_bias = user_biases[users]
        pred = U @ v.T + user_bias
        resid = ratings - pred

        s = precision * np.sum(resid)
        item_bias = s / (precision * nusers + bias_penality)

        resid2 = ratings - user_bias - item_bias

        linsum_u = precision * (U.T @ resid2.T)

        features, _ = get_row_from_csr(features_by_movie_csr, n)
        Fn = len(features)
        linsum_f = vector_penality / np.sqrt(Fn) * np.sum(feature_vectors[features, :], axis=0)
        transition_matrix = vector_penality * np.identity(K) + precision * (U.T @ U)

        item_vector = np.linalg.solve(transition_matrix, linsum_u + with_features * linsum_f)
        return item_bias, item_vector

    for n in prange(N):
        item_biases[n], item_vectors[n, :] = update_item(n)

@njit(parallel=True, fastmath=True)
def als_update_feature_numba_csr_with_features(
    item_vectors: np.ndarray,
    feature_vectors: np.ndarray,
    features_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    movies_by_feature_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    K: int,
) -> None:
    """
    Single ALS iteration with feature regularization.
    
    Args:
        item_vectors: Item embedding matrix (modified in-place).
        features_by_movie_csr: Features by movie CSR matrix.
        movies_by_feature_csr: Movies by feature CSR matrix.
        K: Embedding dimension.
        
    """
    
    F = get_nrows_csr(movies_by_feature_csr)

    def update_feature(i: int):
        items, _ = get_row_from_csr(movies_by_feature_csr, i)
        nitems = len(items)
        Ci = 1
        vect_sum = np.zeros(K)
        
        for nidx in prange(len(items)):
            n = items[nidx]
            features, _ = get_row_from_csr(features_by_movie_csr, n)
            Fn = len(features)
            features_without_i = remove_element(features, i)
            features_without_i_vector = feature_vectors[features_without_i, :]
            mean_feature = 1 / Fn * np.sum(features_without_i_vector, axis=0)
            Ci += 1 / Fn
            vect_sum += item_vectors[n, :] / np.sqrt(Fn) - mean_feature

        feature_vector = 1 / Ci * vect_sum
        return feature_vector

    for i in prange(F):
        feature_vectors[i, :] = update_feature(i)
        
        
@njit(fastmath=True)
def als_iteration_numba_csr_with_features(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    feature_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_item_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    features_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    movies_by_feature_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    vector_penality: float,
    bias_penality: float,
    K: int,
    with_features: bool = False
) -> None:
    """
    Single ALS iteration with feature regularization.
    
    Args:
        user_vectors: User embedding matrix (modified in-place).
        item_vectors: Item embedding matrix (modified in-place).
        user_biases: User bias vector (modified in-place).
        item_biases: Item bias vector (modified in-place).
        ratings_by_user_csr: Ratings by user CSR matrix.
        ratings_by_item_csr: Ratings by item CSR matrix.
        features_by_movie_csr: Features by movie CSR matrix.
        movies_by_feature_csr: Movies by feature CSR matrix.
        precision: Precision parameter.
        vector_penality: Vector regularization parameter.
        bias_penality: Bias regularization parameter.
        K: Embedding dimension.
        with_features: Whether to use item's features.
    """
    als_update_users_numba_csr(
        user_vectors,
        item_vectors,
        user_biases,
        item_biases,
        ratings_by_user_csr,
        precision,
        vector_penality,
        bias_penality,
        K
    )
    als_update_items_numba_csr_with_features(
        user_vectors,
        item_vectors,
        feature_vectors,
        user_biases,
        item_biases,
        ratings_by_item_csr,
        features_by_movie_csr,
        precision,
        vector_penality,
        bias_penality,
        K,
        with_features
    )
    als_update_feature_numba_csr_with_features(
        item_vectors,
        feature_vectors,
        features_by_movie_csr,
        movies_by_feature_csr,
        K,
    )

@show_execution_duration(return_time=True)
def als_update_embeddings_and_biases_numba_csr_with_features(
    ratings_by_user_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_user_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    features_by_movie_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    movies_by_feature_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision: float,
    bias_penality: float,
    vector_penality: float,
    K: int = 10,
    with_features: bool = False,
    max_iter: int = 10,
    feature_penality: float = None,
    is_early_stopping = True,
    patience = 5,
    min_delta = 1e-3,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    Train feature-based ALS model.
    
    Args:
        ratings_by_user_train_csr: Training ratings by user.
        ratings_by_movie_train_csr: Training ratings by movie.
        ratings_by_user_test_csr: Test ratings by user.
        ratings_by_movie_test_csr: Test ratings by movie.
        features_by_movie_csr: Features by movie CSR matrix.
        movies_by_feature_csr: Movies by feature CSR matrix.
        precision: Precision parameter.
        bias_penality: Bias regularization parameter.
        vector_penality: Vector regularization parameter.
        K: Embedding dimension.
        with_features: Whether to use item's features.
        max_iter: Maximum number of iterations.
        feature_penality: Feature initialization variance.
        is_early_stopping:,
        patience:,
        min_delta:,
        seed: Random seed.
    
    Returns:
        ...
    """
    M = get_nrows_csr(ratings_by_user_train_csr)
    N = get_nrows_csr(ratings_by_movie_train_csr)
    F = get_nrows_csr(movies_by_feature_csr)
    
    np.random.seed(seed)
    
    if feature_penality is None:
        feature_penality = vector_penality
        
    fmean = np.zeros(K)
    fcov = (1.0 / np.sqrt(K))  * np.identity(K)
    feature_vectors = np.random.multivariate_normal(fmean, fcov, F)

    umean = np.zeros(K)
    ucov = (1.0 / np.sqrt(K)) * np.identity(K)
    user_vectors = np.random.multivariate_normal(umean, ucov, M)

    item_vectors = np.empty((N, K))
    for n in range(N):
        features, _ = get_row_from_csr(features_by_movie_csr, n)
        Fn = len(features)
        vmean = 1 / np.sqrt(Fn) * np.sum(feature_vectors[features, :], axis=0)
        vcov = (1.0 / np.sqrt(K)) * np.identity(K)
        item_vectors[n, :] = np.random.multivariate_normal(vmean, vcov)

    user_biases = np.zeros(M)
    item_biases = np.zeros(N)

    state = {}
    if with_features:
        mode = "with_features"
    else:
        mode = "with_embeddings"
        
    bias_only = False
    fitted_model = {
        "mode": mode,
        "user_vectors": user_vectors,
        "item_vectors": item_vectors,
        "feature_vectors": feature_vectors,
        "user_biases": user_biases,
        "item_biases": item_biases,
        "precision": precision,
        "bias_penality": bias_penality,
        "vector_penality": vector_penality,
        "feature_penality": feature_penality,
        "K": K,
        "bias_only": bias_only,
        "train_loss_hist": [],
        "train_rmse_hist": [],
        "test_loss_hist": [],
        "test_rmse_hist": [],
        "max_iter": max_iter,
        "nepochs": 0, 
        "best_epoch": state.get('best_epoch'),
        "no_improve": state.get('no_improve')
    }
    fitted_model_best = {
        **fitted_model,
        "train_loss_hist": [],
        "train_rmse_hist": [],
        "test_loss_hist": [],
        "test_rmse_hist": [],
    }
            
    for it in tqdm(range(max_iter), unit="epoch"):
        als_iteration_numba_csr_with_features(
            user_vectors,
            item_vectors,
            feature_vectors,
            user_biases,
            item_biases,
            ratings_by_user_train_csr,
            ratings_by_movie_train_csr,
            features_by_movie_csr,
            movies_by_feature_csr,
            precision,
            vector_penality,
            bias_penality,
            K,
            with_features
        )

        train_rmse, train_loss = compute_metrics_numba_csr_with_features(
            user_vectors,
            item_vectors,
            feature_vectors,
            user_biases,
            item_biases,
            ratings_by_movie_train_csr,
            features_by_movie_csr,
            movies_by_feature_csr,
            precision,
            vector_penality,
            bias_penality,
            with_features=with_features)
        test_rmse, test_loss = compute_metrics_numba_csr_with_features(
            user_vectors,
            item_vectors,
            feature_vectors,
            user_biases,
            item_biases,
            ratings_by_movie_test_csr,
            features_by_movie_csr,
            movies_by_feature_csr,
            precision,
            vector_penality,
            bias_penality,
            with_features=with_features)

        
        fitted_model['train_rmse_hist'].append(train_rmse)
        fitted_model['train_loss_hist'].append(train_loss)
        fitted_model['test_rmse_hist'].append(test_rmse)
        fitted_model['test_loss_hist'].append(test_loss)
        
        fitted_model_best['train_rmse_hist'].append(train_rmse)
        fitted_model_best['train_loss_hist'].append(train_loss)
        fitted_model_best['test_rmse_hist'].append(test_rmse)
        fitted_model_best['test_loss_hist'].append(test_loss)

        should_stop = early_stop(
            state=state,
            score=test_rmse,
            patience=patience,
            min_delta=min_delta,
        )
        if state.get('best_epoch') == it:
            fitted_model_best['user_vectors'] = np.copy(user_vectors)
            fitted_model_best['item_vectors'] = np.copy(item_vectors)
            fitted_model_best['feature_vectors'] = np.copy(feature_vectors)
            fitted_model_best['user_biases'] = np.copy(user_biases)
            fitted_model_best['item_biases'] = np.copy(item_biases)
        
        if (it + 1) % 10 == 0 or it == 0:
            training_info_str = (
                f"epoch {it+1}/{max_iter} | train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} | train_rmse={train_rmse:.4f} "
                f"test_rmse={test_rmse:.4f} |"
                f"best_epoch {state.get('best_epoch')}"
            )
        
            tqdm.write(training_info_str)
            
        if is_early_stopping and should_stop:
            print('early_stopping')
            break
    fitted_model.update({            
        "nepochs": it + 1, 
        "best_epoch": state.get('best_epoch'),
    })
    fitted_model_best.update(
        {
        "nepochs": state.get('best_epoch') + 1,
        "best_epoch": state.get('best_epoch'),
        "no_improve": state.get('no_improve') 
        }
    )
    for k in ['train_rmse_hist', 'train_loss_hist', 'test_rmse_hist', 'test_loss_hist']:
        fitted_model_best[k] = fitted_model_best[k][: fitted_model_best['nepochs']]
    return fitted_model_best, fitted_model


def train_recommender_als(
    ratings_by_user_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_train_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_user_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ratings_by_movie_test_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    features_by_movie_csr:Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]=None,
    movies_by_feature_csr:Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]=None,
    training_function: Callable = None,
    vector_penality: float = 1e-2,
    precision: float = 1,
    bias_penality: float = 1e-3,
    K: int = 8,
    max_iter: int = 10,
    model_folder: str = 'models',
    data_size_str: str = '',
    bias_only: bool = False,
    with_features: bool = False,
    items_ids:List[int] = None,
    feature_ids:List[str] = None,
    save: bool = False,
    seed: int = 42,
    training_function_kwargs: Optional[Dict[str, Any]] = {}
) -> Dict[str, Any]:
    """
    High-level wrapper for training ALS recommender system.
    
    This function handles the complete training workflow including:
    - Training the model (bias-only or full matrix factorization)
    - Plotting training curves
    - Saving models with timestamps
    - Logging training runs to YAML
    
    Args:
        ratings_by_user_train_csr: Training ratings by user CSR matrix.
        ratings_by_movie_train_csr: Training ratings by movie CSR matrix.
        ratings_by_user_test_csr: Test ratings by user CSR matrix.
        ratings_by_movie_test_csr: Test ratings by movie CSR matrix.
        features_by_movie_csr: Features by movie CSR matrix.
        movies_by_feature_csr: Movies by feature CSR matrix.
        vector_penality: Vector regularization parameter (tau).
        precision: Precision parameter (lambda).
        bias_penality: Bias regularization parameter.
        K: Embedding dimension.
        max_iter: Maximum number of iterations.
        model_folder: Root folder for saving models.
        data_size_str: Dataset identifier (e.g., 'ml-32m').
        bias_only: If True, train bias-only model.
        items_ids: items ids.
        feature_ids: features ids.
        save: If True, save model and logs.
        seed: Random seed for reproducibility.
        training_function_kwargs: Additional kwargs for training function.
    
    Returns:
        Dictionary containing trained model parameters and metadata.
    """
    if features_by_movie_csr is None and movies_by_feature_csr is None:
        with_features = False
        

    if with_features:
        bias_only = False
        mode = "with_features"
        
    elif bias_only:
        mode = "bias_only"
    else:  
        mode = "with_embeddings"
    
    model_folder = Path(model_folder)
    save_path_folder = model_folder / mode
    
    def fun_save(
        fitted_model: Dict[str, Any],
        save_path_folder,
        )-> None:
        """Save model to disk and log to YAML."""
        model_filename = f"{data_size_str}-als.joblib"
        model_filename = add_suffix_model_info(model_filename, fitted_model)
        save_path = save_path_folder / model_filename
                
        save_path = Path(save_path)
        save_path_latest = save_path.with_stem(save_path.stem + "-latest")
        
        suffix = f"iter{max_iter}-{mode}"

        log_file = model_folder / f"als_runs_{suffix}.yaml"
        log_file = log_file.as_posix()
        
        args = {
            "precision": fitted_model.get("precision"),
            "bias_penality": fitted_model.get("bias_penality"),
            "vector_penality": fitted_model.get("vector_penality"),
            "K": fitted_model.get("K"),
            "bias_only": fitted_model.get("bias_only"),
            "train_rmse_hist": fitted_model.get("train_rmse_hist", [])[-2:],
            "test_rmse_hist": fitted_model.get("test_rmse_hist", [])[-2:],
            "nepochs": fitted_model.get("nepochs"),
            "max_iter": fitted_model.get("max_iter"),
            "timestamp": fitted_model.get("timestamp"),
            "model_filename": model_filename,
        }

        saved_path = dump_objects(
            fitted_model,
            path=save_path,
            timestamp=True
        )

        run_id, log_file = log_als_run_to_yaml(log_file, args)

        print("Logged run:", run_id)
        print("YAML log file:", log_file)
        saved_path_latest = dump_objects(
            fitted_model,
            path=save_path_latest,
            timestamp=False
        )
    
    print("Starting training process...")
    is_early_stopping = training_function_kwargs.get('is_early_stopping', False)
    patience = training_function_kwargs.get('patience', 5)
    min_delta = training_function_kwargs.get('min_delta', 1e-3)
    
    if mode == 'bias_only':
        training_function = als_update_biases_numba_csr_bias_only
        print(f"Training model(bias only), K:{K}")
        (
            (user_biases, item_biases, train_loss_hist,
            train_rmse_hist, test_loss_hist, test_rmse_hist), 
            training_starting_time, training_duration
        ) = training_function(
            ratings_by_user_train_csr,
            ratings_by_movie_train_csr,
            ratings_by_user_test_csr,
            ratings_by_movie_test_csr,
            precision,
            bias_penality,
            K,
            max_iter=max_iter,
            is_early_stopping = is_early_stopping,
            patience = patience,
            min_delta = min_delta,
        )
    elif mode == 'with_embeddings':
        training_function = als_update_embeddings_and_biases_numba_csr
        print(f"Training model(with embeddings), K:{K}, lambda:{precision}, tau:{vector_penality}")
        
        (
            (fitted_model_best, fitted_model), training_starting_time, training_duration
        ) = training_function(
            ratings_by_user_train_csr,
            ratings_by_movie_train_csr,
            ratings_by_user_test_csr,
            ratings_by_movie_test_csr,
            precision,
            bias_penality,
            vector_penality,
            K,
            max_iter=max_iter,
            is_early_stopping = is_early_stopping,
            patience = patience,
            min_delta = min_delta,
            seed=seed
        )
    elif mode == 'with_features':
        bias_only = False
        training_function = als_update_embeddings_and_biases_numba_csr_with_features
        feature_penality = training_function_kwargs.get('feature_penality', vector_penality)
        print(f"Training model(with features), K:{K}, lambda:{precision}, tau:{vector_penality}")
        (
            (fitted_model_best, fitted_model), training_starting_time, training_duration
        ) = training_function(
            ratings_by_user_train_csr,
            ratings_by_movie_train_csr,
            ratings_by_user_test_csr,
            ratings_by_movie_test_csr,
            features_by_movie_csr,
            movies_by_feature_csr,
            precision,
            bias_penality,
            vector_penality,
            K,
            with_features=True,
            max_iter=max_iter,
            feature_penality=feature_penality,
            is_early_stopping = is_early_stopping,
            patience = patience,
            min_delta = min_delta,
            seed=seed,
        )
    
    plot_als_curves(
        fitted_model.get('train_loss_hist'),
        fitted_model.get('test_loss_hist'),
        fitted_model.get('train_rmse_hist'),
        fitted_model.get('test_rmse_hist'),
        fitted_model.get('K'),
        fitted_model.get('precision'),
        fitted_model.get('vector_penality'),
        fitted_model.get('bias_penality'),
        bias_only=bias_only
    )
    nratings_by_item_train = np.array(get_nelements_by_row_csr(ratings_by_movie_train_csr))
    nratings_by_item_test = np.array(get_nelements_by_row_csr(ratings_by_movie_test_csr))
    nratings_by_item = nratings_by_item_train + nratings_by_item_test
    
    update_dict = {
        "items_ids":items_ids,
        "features_ids": feature_ids,
        "nratings_by_item": nratings_by_item, 
        "training_starting_time": training_starting_time,
        "training_duration": training_duration,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    fitted_model.update(update_dict)
    fitted_model_best.update(update_dict)
    if save:
        fun_save(fitted_model_best, save_path_folder)
        if not is_early_stopping:
            fun_save(fitted_model, save_path_folder)
        
    return fitted_model_best


def vectors_norm(vectors: np.ndarray) -> np.ndarray:
    """
    Compute L2 norms of each row in a 2D array of vectors.

    Args:
        vectors (np.ndarray): 2D array of shape (n_items, n_features).

    Returns:
        np.ndarray: 1D array of L2 norms of each vector.
    """
    return np.linalg.norm(vectors, axis=1)


def get_top_long_trait_vector(
    vectors: np.ndarray,
    sort_flag: bool = False,
    topn: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get L2 norms of item vectors and optionally return the top vectors sorted by length.

    Args:
        vectors (np.ndarray): 2D array of shape (nitems, nfeatures).
        sort_flag (bool): Whether to sort vectors by their norm in descending order.
        topn (Optional[int]): Number of top vectors to return. If None, return all.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Norms of the vectors (sorted if requested).
            - Indices corresponding to the sorted norms.
    """
    norms = vectors_norm(vectors)
    nitems = len(norms)

    if topn is not None and topn < nitems:
        top_indices = np.argpartition(-norms, topn - 1)[:topn]
        top_norms = norms[top_indices]
    else:
        top_indices = np.arange(nitems)
        top_norms = norms

    if sort_flag:
        sorted_order = np.argsort(-top_norms)
        top_indices = top_indices[sorted_order]
        top_norms = top_norms[sorted_order]

    return top_norms, top_indices