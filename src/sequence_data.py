from __future__ import annotations

import bisect
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


class PickleSequenceDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.samples = load_pickle(self.path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


class SequenceShardDataset:
    def __init__(self, manifest_path: str | Path, cache_size: int = 4) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        self.base_dir = self.manifest_path.parent
        self.cache_size = max(1, cache_size)
        self.shards = self.manifest["shards"]
        self.cumulative_counts: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative_counts.append(total)

        self._cache: OrderedDict[int, list[tuple[list[int], int]]] = OrderedDict()

    def __len__(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        shard_index = bisect.bisect_right(self.cumulative_counts, idx)
        previous_total = 0 if shard_index == 0 else self.cumulative_counts[shard_index - 1]
        local_index = idx - previous_total
        return self._load_shard(shard_index)[local_index]

    def _load_shard(self, shard_index: int) -> list[tuple[list[int], int]]:
        if shard_index in self._cache:
            self._cache.move_to_end(shard_index)
            return self._cache[shard_index]

        shard_path = self.base_dir / self.shards[shard_index]["path"]
        samples = load_pickle(shard_path)
        self._cache[shard_index] = samples
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return samples


class SliceDataset:
    def __init__(self, dataset, start: int = 0, end: int | None = None) -> None:
        self.dataset = dataset
        self.start = max(0, start)
        self.end = len(dataset) if end is None else min(end, len(dataset))
        if self.end < self.start:
            self.end = self.start

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.dataset[self.start + idx]


def load_sequence_dataset(root: str | Path, cache_size: int = 4, data_dir: str | Path | None = None):
    root = Path(root)
    data_path = Path(data_dir) if data_dir else root / "data"
    if not data_path.is_absolute():
        data_path = root / data_path
    manifest_path = data_path / "sequences_manifest.json"
    legacy_path = data_path / "sequences.pkl"
    if manifest_path.exists():
        return SequenceShardDataset(manifest_path, cache_size=cache_size)
    if legacy_path.exists():
        return PickleSequenceDataset(legacy_path)
    raise FileNotFoundError(f"Run `python src/preprocess.py` before training. Missing data in {data_path}.")
