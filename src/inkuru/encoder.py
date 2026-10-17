"""AfroXLMR-mini helpers shared by notebooks 03 and 04: the dataset, batching, device, and prediction."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from inkuru.paths import MAX_LENGTH


class TextSet(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate(tokenizer, batch):
    texts, labels = zip(*batch)
    encoded = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Softmax scores and gold indices for every batch in the loader."""
    model.eval()
    probs, golds = [], []
    for batch in loader:
        golds.append(batch.pop("labels"))
        batch = {key: value.to(device) for key, value in batch.items()}
        probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu())
    return torch.cat(probs).numpy(), torch.cat(golds).numpy()
