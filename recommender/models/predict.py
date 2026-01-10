from os.path import exists
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from typing import Any, Tuple, List, Optional,  Dict
from IPython.display import display
import pandas as pd
from ..utils.helpers import prepend_attrs_to_latex, argsort_array
from ..data.csr import get_nrows_csr
from ..data.ratings import (
    get_id_from_idx,
    get_id_from_idx,
    get_movie_info_from_id,
    get_movie_title_from_id,
    get_features_id_from_movie_id
    )
from .train_before_scheduler import als_update_users_numba_csr

def get_user_embedding(
    ratings_by_user: List[List[Tuple[int, float]]],
    item_vectors: np.ndarray,
    item_biases: np.ndarray,
    precision: float,
    bias_penality: float,
    vector_penality: float,
    max_iter: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute user embeddings from their ratings.
    
    Args:
        ratings_by_user: List of [(item_idx, rating)] for each user.
        item_vectors: Item embedding matrix.
        item_biases: Item bias vector.
        precision: Precision parameter.
        bias_penality: Bias regularization parameter.
        vector_penality: Vector regularization parameter.
        max_iter: Number of ALS iterations.
    
    Returns:
        Tuple of (user_biases, user_vectors).
    """
    M = len(ratings_by_user)
    K = item_vectors.shape[1]
    mean = np.zeros(K)
    cov = (1.0 / np.sqrt(K)) * np.identity(K)
    user_vectors = np.random.multivariate_normal(mean, cov, M)
    user_biases = np.zeros(M)
    
    def update_user(m: int = 0):
        items, ratings = zip(*ratings_by_user[m])
        items = list(items)
        ratings = list(ratings)
        nitems = len(items)
        if nitems == 0:
            return user_biases[m], user_vectors[m]

        V = item_vectors[items]
        u = user_vectors[m]
        item_bias = item_biases[items]
        user_bias = user_biases[m]
        pred = V @ u + item_bias
        resid = ratings - pred

        s = precision * np.sum(resid)
        user_bias = s / (precision * nitems + bias_penality)

        resid2 = ratings - user_bias - item_bias

        linsum_v = precision * (resid2 @ V)
        transition_matrix = vector_penality * np.identity(K) + precision * (V.T @ V)

        user_vector = np.linalg.solve(transition_matrix, linsum_v)
        return user_bias, user_vector
    
    for it in tqdm(range(max_iter)):
        for m in range(M):
            user_biases[m], user_vectors[m, :] = update_user(m)

    return user_biases, user_vectors

def get_user_embedding_csr(
    ratings_by_user_csr: Tuple[np.ndarray, np.ndarray, np.ndarray],
    item_vectors: np.ndarray,
    item_biases: np.ndarray,
    precision: float,
    bias_penality: float,
    vector_penality: float,
    max_iter: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute user embeddings from their ratings.
    
    Args:
        ratings_by_user_train_csr: ratings by user.
        item_vectors: Item embedding matrix.
        item_biases: Item bias vector.
        precision: Precision parameter.
        bias_penality: Bias regularization parameter.
        vector_penality: Vector regularization parameter.
        max_iter: Number of ALS iterations.
    
    Returns:
        Tuple of (user_biases, user_vectors).
    """
    M = get_nrows_csr(ratings_by_user_csr)
    K = item_vectors.shape[1]
    mean = np.zeros(K)
    cov = (1.0 / np.sqrt(K)) * np.identity(K)
    user_vectors = np.random.multivariate_normal(mean, cov, M)
    user_biases = np.zeros(M)
    
    for it in tqdm(range(max_iter), unit="epoch"):
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
    return user_biases, user_vectors
   
def get_score_from_users_embedding(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    bias_factor: float = 0.05
) -> np.ndarray:
    """
    Calculate predicted scores for all user-item pairs.
    
    Args:
        user_vectors: User embedding matrix (M x K).
        item_vectors: Item embedding matrix (N x K).
        user_biases: User bias vector (M,).
        item_biases: Item bias vector (N,).
        bias_factor: Scaling factor for item biases.
    
    Returns:
        Score matrix (M x N).
    """
    M = len(user_vectors)
    N = len(item_vectors)
    items_score_by_user = np.empty((M, N))
    for m in range(M):
        items_score_by_user[m, :] = item_vectors @ user_vectors[m].T + bias_factor * item_biases
    return items_score_by_user




def get_top_recommended_item(
    ratings_by_user: List[List[Tuple[int, float]]],
    movie_idx_to_id: List[int],
    item_vectors: np.ndarray,
    item_biases: np.ndarray,
    precision: float,
    bias_penality: float,
    vector_penality: float,
    max_iter: int = 50,
    ntop: int = -1,
    bias_factor: float = 1e-3,
    nratings_by_item: Optional[np.ndarray]=None,
    atleast_nratings:int= 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top N recommended items for each user.
    
    Args:
        ratings_by_user: List of [(item_idx, rating)] for each user.
        movie_idx_to_id: Mapping from movie index to movie ID.
        item_vectors: Item embedding matrix.
        item_biases: Item bias vector.
        precision: Precision parameter.
        bias_penality: Bias regularization parameter.
        vector_penality: Vector regularization parameter.
        max_iter: Number of iterations for computing user embeddings.
        ntop: Number of top items to return (-1 for all).
        bias_factor: Scaling factor for item biases.
        nratings_by_item: Array of number of ratings per item.
        atleast_nratings: Minimum number of ratings an item must have to be considered.
    
    Returns:
        Tuple of (top_item_ids, top_scores) arrays of shape (M, ntop).
    """
    M = len(ratings_by_user)
    N = len(item_vectors)
    
    user_biases, user_vectors = get_user_embedding(
        ratings_by_user,
        item_vectors,
        item_biases,
        precision,
        bias_penality,
        vector_penality,
        max_iter,
    )
    
    if ntop == -1:
        ntop = N
    
    ntop_score_by_user = np.empty((M, ntop), dtype=float)
    ntop_item_ids_by_user = np.empty((M, ntop), dtype=int)

    items_score_by_user = get_score_from_users_embedding(
        user_vectors, item_vectors,
        user_biases, item_biases,
        bias_factor=bias_factor
    )

    if atleast_nratings is not None and nratings_by_item is not None:
        ignored_item_mask = nratings_by_item < atleast_nratings
        items_score_by_user[:, ignored_item_mask] = -np.inf
    for m in range(len(items_score_by_user)):
        scores = items_score_by_user[m]
        ntop_indices = argsort_array(scores, descending=True)[:ntop]
        ntop_item_ids_by_user[m, :] = get_id_from_idx(ntop_indices, movie_idx_to_id)
        ntop_score_by_user[m, :] = scores[ntop_indices]

    return ntop_item_ids_by_user, ntop_score_by_user

def get_save_top_recommended(
    fitted_model: Dict[str, Any],
    ratings_by_user: List[List[Tuple[int, float]]],
    movie_idx_to_id: Dict[int, int],
    movie_id_to_idx: Dict[int, int],
    feature_idx_to_id: Dict[int, int],
    movies_title: Dict[int, str],
    features_by_movie_csr,
    ntop: int,
    max_iter: int,
    atleast_nratings: int = None,
    bias_factors: List[float] = [1.0, 1e-2, 1e-3],
    save_to_latex:bool= True,
    folder_save_path:str = 'prediction',
) -> List[pd.DataFrame]:
    """
    Compute top-N movie recommendations for a user under different bias
    scaling factors and return them as pandas DataFrames.

    Each returned DataFrame contains movie metadata for the top
    recommendations and stores model and run configuration in
    `DataFrame.attrs`.

    Args:
        fitted_model:
            Dictionary containing trained model parameters.
        ratings_by_user:
            List[List[]] of (item_idx, rating) pairs representing users input.
        movie_idx_to_id:
            Mapping from internal movie indices to movie IDs.
        movie_id_to_idx:
            Mapping from movie IDs to internal movie indices.
        feature_idx_to_id:
            Mapping from feature indices to feature IDs.
        movies_title:
            Mapping from movie IDs to movie titles.
        features_by_movie_csr:
            CSR matrix of movie–feature associations.
        ntop:
            Number of top items to recommend.
        max_iter:
            Maximum number of optimization iterations.
        atleast_nratings:
            Minimum number of ratings required for an item to be considered.
        bias_factors:
            Scaling factors applied to item bias during recommendation.

    Returns:
        A list of pandas DataFrames, one per bias factor, containing the
        top-N recommended movies.
    """
    dfs = []

    for bias_factor in bias_factors:
        recommended_movies_ids, recommended_movie_score  = get_top_recommended_item(
                ratings_by_user,
                movie_idx_to_id,
                fitted_model['item_vectors'],
                fitted_model['item_biases'],
                fitted_model['precision'],
                fitted_model['bias_penality'],
                fitted_model['vector_penality'],
                max_iter=max_iter,
                ntop=ntop,
                bias_factor=bias_factor,
                nratings_by_item= fitted_model['nratings_by_item'],
                atleast_nratings=atleast_nratings
            )

        recommended_info = []
        for user_recommended_movie_ids in recommended_movies_ids:
            recommended_movie_info = get_movie_info_from_id(
            user_recommended_movie_ids, movie_id_to_idx,
            feature_idx_to_id,
            movies_title,
            features_by_movie_csr)
            recommended_info.append(recommended_movie_info)
        print(
            f"Model training Info {fitted_model['mode']}, "
            f"K={fitted_model['K']}, "
            f"precision={fitted_model['precision']}, "
            f"bias_penality={fitted_model['bias_penality']}, "
            f"vector_penality={fitted_model['vector_penality']}, "
            f"bias_factor={bias_factor}"
        )

        print("Movies recommended to user rating the movie:")
        movie_rated = []
        for idx, rating in ratings_by_user[0]:
            id = get_id_from_idx(idx, movie_idx_to_id)
            title = get_movie_title_from_id(id, movie_id_to_idx, movies_title)
            features = get_features_id_from_movie_id(
                id, movie_id_to_idx,
                feature_idx_to_id,
                features_by_movie_csr
                )
            movie_rated.append((title, features, rating))
            print(f"- {title}: {features} (rating: {rating})")
        df = pd.DataFrame(recommended_info[0])
        df.attrs = {
            "model_info": {
                "mode": fitted_model['mode'],
                "K": fitted_model['K'],
                "precision": fitted_model['precision'],
                "bias_penality": fitted_model['bias_penality'],
                "vector_penality": fitted_model['vector_penality']
            },
            "bias_factor": bias_factor,
            "movies_rated": movie_rated

        }
        if atleast_nratings is not None:
            atleast_suffix = f"filtered_atleast{atleast_nratings}_ratings"
        else:
            atleast_suffix = ''
        display(df)
        dfs.append(df)
    if save_to_latex:
        folder_save_path  = Path(folder_save_path )
        folder_save_path.mkdir(parents=True, exist_ok=True)
        for df in dfs:
            filename = f"predict_{fitted_model['mode']}-bias_factor={df.attrs['bias_factor']}{atleast_suffix}.tex"
            prediction_latex_table_path = folder_save_path / filename
            df.to_latex(prediction_latex_table_path)
            prepend_attrs_to_latex(
                prediction_latex_table_path,
                df.attrs
                )
            print(f"prediction saved with the name {filename}")
    return dfs

