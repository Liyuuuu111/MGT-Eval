
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Union
import json, os
import pandas as pd

Example = Dict[str, Any]

def _from_dataframe(df: "pd.DataFrame") -> Iterator[Example]:
    for _, row in df.iterrows():
        yield {"text": row["text"], "label": int(row["label"])}

def _from_list_of_dict(lst: Sequence[Dict[str, Any]]) -> Iterator[Example]:
    for ex in lst:
        assert "text" in ex and "label" in ex, "Each item must have 'text' and 'label'"
        yield {"text": ex["text"], "label": int(ex["label"])}

def load_jsonl(path: str) -> List[Example]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def ensure_iter_dataset(dataset: Union[str, Iterable[Example], "pd.DataFrame", Sequence[Dict[str, Any]]]) -> Iterable[Example]:
    """
    Accept:
      - path to .jsonl with {"text": ..., "label": 0/1}
      - pandas DataFrame with columns ['text', 'label']
      - list/tuple of dicts with keys 'text', 'label'
      - any iterable of such dicts
    Returns an iterable over {"text", "label"}.
    """
    if isinstance(dataset, str) and os.path.isfile(dataset) and dataset.endswith(".jsonl"):
        lst = load_jsonl(dataset)
        return _from_list_of_dict(lst)
    try:
        import pandas as pd  # type: ignore
        if isinstance(dataset, pd.DataFrame):
            return _from_dataframe(dataset)
    except Exception:
        pass
    if isinstance(dataset, (list, tuple)):
        return _from_list_of_dict(dataset)
    # Assume it's already an iterable of dicts
    return dataset

def split_dataset(
    data: Sequence[Example], train_ratio: float = 0.8, seed: int = 42
) -> (List[Example], List[Example]):
    import random
    data = list(data)
    rng = random.Random(seed)
    rng.shuffle(data)
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]
