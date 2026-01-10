import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Any, Dict, Union, Tuple, List, Iterable, Hashable
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import warnings
import urllib.request
from functools import wraps
import shutil
import re
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def is_kaggle() -> bool:
    """
    Detect if the code is running on Kaggle.
    
    Returns:
        bool: True if running on Kaggle, False otherwise.
    """
    return "KAGGLE_URL_BASE" in os.environ


def is_colab() -> bool:
    """
    Detect if the code is running on Google Colab.
    
    Returns:
        bool: True if running on Google Colab, False otherwise.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def set_project(
    fun_todo_kaggle: Optional[Callable[..., Any]] = None,
    fun_todo_colab: Optional[Callable[..., Any]] = None,
    fun_todo_local: Optional[Callable[..., Any]] = None,
    kwargs_kaggle: Optional[Dict[str, Any]] = None,
    kwargs_colab: Optional[Dict[str, Any]] = None,
    kwargs_local: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Execute platform-specific setup functions based on the current environment.
    
    This function detects whether code is running on Kaggle, Google Colab, or locally,
    and executes the corresponding setup function with the provided keyword arguments.
    
    Args:
        fun_todo_kaggle: Optional function to execute on Kaggle.
        fun_todo_colab: Optional function to execute on Google Colab.
        fun_todo_local: Optional function to execute locally.
        kwargs_kaggle: Keyword arguments to pass to the Kaggle function.
        kwargs_colab: Keyword arguments to pass to the Colab function.
        kwargs_local: Keyword arguments to pass to the local function.
    
    Returns:
        The return value of the executed function, or None if no function is provided
        for the detected environment.
    """
    kwargs_kaggle = kwargs_kaggle or {}
    kwargs_colab = kwargs_colab or {}
    kwargs_local = kwargs_local or {}
    
    if is_kaggle():
        print("Running on Kaggle")
        if fun_todo_kaggle is not None:
            return fun_todo_kaggle(**kwargs_kaggle)
    elif is_colab():
        print("Running on Google Colab")
        if fun_todo_colab is not None:
            return fun_todo_colab(**kwargs_colab)
    else:
        print("Running locally")
        if fun_todo_local is not None:
            return fun_todo_local(**kwargs_local)
    
    return None

        
def show_execution_duration(
    info: str = "",
    return_time: bool = False
) -> Callable[
    [Callable[P, R]],
    Callable[P, Union[R, Tuple[R, datetime, timedelta]]]
]:
    """
    Decorator factory that prints execution duration and timestamps of a function.
    
    This decorator prints the function name, start time, end time, and total duration
    of the decorated function's execution. Optionally, it can return the execution
    start time and duration along with the function's result.
    
    Args:
        info: Optional information string to print before execution.
        return_time: If True, returns a tuple of (result, start_time, duration).
                    If False, returns only the result.
    
    Returns:
        A decorator function that wraps the target function with execution timing.
    """
    if len(info):
        print(f"{info}")
    
    def inner_fun(
        fun: Callable[P, R]
    ) -> Callable[P, Union[R, Tuple[R, datetime, timedelta]]]:
        @wraps(fun)
        def wrapper(
            *args: P.args,
            **kwargs: P.kwargs
        ) -> Union[R, Tuple[R, datetime, timedelta]]:
            print(f"Execution of:  {fun.__name__}")

            start = datetime.now()
            print(f"Start at: {start.strftime('%Y-%m-%d_%H-%M-%S')}")

            result = fun(*args, **kwargs)

            end = datetime.now()
            print(f"Finished at: {end.strftime('%Y-%m-%d_%H-%M-%S')}")

            duration = end - start
            print(f"Execution in: {duration}")

            if return_time:
                return result, start, duration
            return result

        return wrapper

    return inner_fun


def dump_objects(
    mobject: Any,
    path: Union[str, Path] = "models/data.joblib",
    compress: int = 3,
    timestamp: bool = False,
) -> str:
    """
    Save Python objects to disk using joblib compression.
    
    Args:
        mobject: The object(s) to save.
        path: File path where the object will be saved.
        compress: Compression level (0-9, default 3).
        timestamp: If True, append timestamp to filename.
    
    Returns:
        The final path where the object was saved.
    """
    path = str(path)
    
    if timestamp:
        base, ext = os.path.splitext(path)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path = f"{base}_{current_time}{ext}"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(mobject, path, compress=compress)
    print(f"Objects successfully saved to '{path}'")
    return path


def load_objects(path: Union[str, Path] = "models/data.joblib") -> Any:
    """
    Load Python objects from disk using joblib.
    
    Args:
        path: File path from which to load the object.
    
    Returns:
        The loaded object(s).
    """
    objs = load(str(path))
    return objs


def search_regex(
    lst: List[str], 
    pattern: str, 
    flags: int = 0
) -> List[int]:
    """
    Search for regex pattern in a list of strings and return matching indices.
    
    Args:
        lst: List of strings to search through.
        pattern: Regular expression pattern to match.
        flags: Regex flags (e.g., re.IGNORECASE).
    
    Returns:
        List of indices where the pattern was found.
    """
    regex = re.compile(pattern, flags)
    return [i for i, s in enumerate(lst) if regex.search(s)]


def get_paths(folder: Union[str, Path], pattern: str) -> List[Path]:
    """
    Return list of file paths in folder matching the given pattern.
    
    Args:
        folder: Directory to search in.
        pattern: Regular expression pattern to match against filenames.
    
    Returns:
        List of Path objects matching the pattern.
    """
    folder = Path(folder)
    regex = re.compile(pattern)
    return [p for p in folder.iterdir() if regex.match(p.name)]


def ensure_float_list(v: Any) -> List[float]:
    """
    Convert scalars, numpy scalars, arrays to clean Python float list.
    
    Args:
        v: Value to convert (scalar, numpy array, or list).
    
    Returns:
        List of floats.
    """
    if isinstance(v, (int, float, np.generic)):
        return [float(v)]
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.tolist()]
    if isinstance(v, list):
        return [float(x) for x in v]
    return [float(v)]


def sanitize_value(v: Any) -> Any:
    """
    Convert types YAML can't handle to simple strings/floats.
    
    Args:
        v: Value to sanitize.
    
    Returns:
        Sanitized value that can be serialized to YAML.
    """
    if isinstance(v, datetime):
        return v.isoformat(timespec="seconds")
    if isinstance(v, timedelta):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [sanitize_value(x) for x in v.tolist()]
    return v


def sanitize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply sanitize_value recursively to all entries in a dictionary.
    
    Args:
        entry: Dictionary to sanitize.
    
    Returns:
        Sanitized dictionary with YAML-compatible values.
    """
    out = {}
    for k, v in entry.items():
        if isinstance(v, dict):
            out[k] = sanitize_entry(v)
        elif isinstance(v, list):
            out[k] = [sanitize_value(x) for x in v]
        else:
            out[k] = sanitize_value(v)
    return out


def strip_timestamp_suffix(path: Path) -> Path:
    """
    Remove timestamp suffix from a filename, preserving extension.
    
    Args:
        path: File path with potential timestamp.
    
    Returns:
        Path with timestamp removed.
    """
    
    TIMESTAMP_REGEX = re.compile(
        r"20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}(-\d{2})?(-\d{1,3})?"

    )
    stem = path.stem
    new_stem = TIMESTAMP_REGEX.sub('', stem)
    return path.with_name(new_stem + path.suffix)


def ensure_unique_path(path: Path) -> Path:
    """
    Avoid overwriting files by appending _1, _2, etc.
    
    Args:
        path: Desired file path.
    
    Returns:
        Unique file path.
    """
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    counter = 1

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def clean_filenames_recursive(folder: str) -> None:
    """
    Recursively clean all filenames in a folder by removing timestamp suffixes.
    
    Args:
        folder: Path to the root folder.
    """
    folder = Path(folder)

    for file in folder.rglob("*"):
        if file.is_file():
            new_path = strip_timestamp_suffix(file)

            if new_path != file:
                new_path = ensure_unique_path(new_path)
                file.rename(new_path)
                print(f"Renamed: {file.name} → {new_path.name}")

    print("Done! All filenames cleaned.")
    


def download_file(
    url: str,
    output_dir: str,
    filename: Optional[str] = None,
    overwrite: bool = False
) -> Path:
    """
    Download a file from a URL to a local directory.

    Args:
        url: URL of the file to download.
        output_dir: Directory where the file is saved.
        filename: Name of the saved file. If None, infer from URL.
        overwrite: Whether to overwrite an existing file.

    Returns:
        Path to the downloaded file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = Path(url).name

    file_path = output_path / filename

    if file_path.exists() and not overwrite:
        return file_path

    urllib.request.urlretrieve(url, file_path)

    return file_path


def copy_folder(
    src: Union[str, Path],
    dst: Union[str, Path],
    include_patterns: Union[str, List[str]] = ".*",
    exclude_patterns: Union[str, List[str]] = None,
):
    """
    Copy files from `src` to `dst` matching include regex patterns
    and not matching exclude regex patterns.

    Args:
        src (str or Path): Source folder path.
        dst (str or Path): Destination folder path.
        include_patterns (str or list of str, optional): Regex pattern(s) of files to include.
            Default is '.*' (all files). Example: r".*\.txt$" or [r".*\.txt$", r".*\.py$"].
        exclude_patterns (str or list of str, optional): Regex pattern(s) of files to exclude.
            Default is None. Example: r"ignore_.*\.txt$" or [r"ignore_.*\.txt$", r"temp_.*\.py$"].
    """
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    if isinstance(include_patterns, str):
        include_patterns = [include_patterns]
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = [exclude_patterns]

    for file in src_path.rglob("*"):
        if not file.is_file():
            continue

        rel_path_str = str(file.relative_to(src_path))

        if not any(search_regex([rel_path_str], pat) for pat in include_patterns):
            continue

        if any(search_regex([rel_path_str], pat) for pat in exclude_patterns):
            continue

        target_file = dst_path / rel_path_str
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target_file)

def prepend_attrs_to_latex(
    tex_path: str,
    header: Union[str, dict[str, Any], list[Any], tuple[Any, ...]],
    prefix: str = "% ",
    encoding: str = "utf-8"
) -> None:
    """Prepend formatted header attributes to the beginning of a LaTeX file.
    
    Reads a LaTeX file and inserts formatted header lines at the top. The header
    can be a dictionary (formatted as key-value pairs), a list/tuple (formatted
    as individual items), or a string (used as-is). Header lines are prefixed
    with a specified prefix (default: "% " for LaTeX comments).
    
    Args:
        tex_path: Path to the LaTeX file to modify.
        header: Header content to prepend. Can be a string, dict with key-value
            pairs, or list/tuple of items to format.
        prefix: String prefix to add before each header line. Defaults to "% "
            for LaTeX comment syntax.
        encoding: Text encoding for reading/writing the file. Defaults to "utf-8".
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If the specified tex_path does not exist.
    """
    path = Path(tex_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not header:
        return
    
    if isinstance(header, dict):
        lines = [f"{prefix}{k}: {v}" for k, v in header.items()]
    elif isinstance(header, (list, tuple)):
        lines = [f"{prefix}{item}" for item in header]
    else:
        lines = [f"{prefix}{header}"]
    
    block = "\n".join(lines) + "\n\n"
    content = path.read_text(encoding=encoding)
    
    if content.startswith(lines[0]):
        return
    
    path.write_text(block + content, encoding=encoding)

def get_indices(
    values: Iterable[Hashable],
    items: Iterable[Hashable]
) -> List[Optional[int]]:
    """
    Return the indices of values within a reference list.

    Each element in `values` is looked up in `items`, and its
    corresponding index is returned. If a value is not found,
    None is returned in its place.

    Args:
        values: Iterable of values to look up.
        items: Reference iterable containing all possible values.

    Returns:
        A list of indices corresponding to `values`, preserving order.
        Missing values are represented as None.

    """
    index_map = {item: i for i, item in enumerate(items)}
    return [index_map.get(v) for v in values]

def argsort_array(arr: np.ndarray, descending: bool = False) -> np.ndarray:
    """
    Get indices that would sort an array.
    
    Args:
        arr: Array to sort.
        descending: If True, sort in descending order.
    
    Returns:
        Array of sorted indices.
    """
    sort_indices = np.argsort(arr)
    if descending:
        sort_indices = np.flip(sort_indices)
    return sort_indices


def sort_array(
    items: np.ndarray,
    descending: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort a 1D array-like object and return sorted values with original indices.

    Args:
        items: One-dimensional Python List or NumPy array.
        descending: If True, sort in descending order.

    Returns:
        Tuple(sorted_items, sorted_indices):
    """
    arr = np.asarray(items)

    sorted_indices = argsort_array(arr, descending=descending)
    sorted_items = arr[sorted_indices]

    return sorted_items, sorted_indices

def get_random_vectors(
    arr: np.ndarray,
    sample_size: int = 100,
    axis: int = 0,
    seed: Optional[int] = 42
) ->Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample along a specified axis of a NumPy array.

    Args:
        arr (np.ndarray): Input array (1D or 2D).
        sample_size (int, optional): Number of elements (rows or columns) to sample. Default is 100.
        axis (int, optional): Axis along which to sample. 0 = rows, 1 = columns. Default is 0.
        seed (Optional[int], optional): Random seed for reproducibility. Default is 42.

    Returns:
       Tuple(sampled rows or columns, sorted_indices): Array containing the sampled rows or columns.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = arr.shape[axis]
    if sample_size > n:
        raise ValueError(f"sample_size ({sample_size}) cannot exceed size along axis {axis} ({n})")
    
    indices = np.random.choice(n, size=sample_size, replace=False)
    
    if axis == 0:
        return arr[indices, ...], indices
    elif axis == 1:
        return arr[:, indices], indices
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

