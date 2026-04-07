import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import load_dataset


class ActivationsStore:
    def __init__(self, model: HookedRootModule, cfg: dict, dataset=None):
        """Create an ActivationsStore.

        Args:
            model: A TransformerLens HookedTransformer.
            cfg: Configuration dict.  Must contain at minimum:
                ``hook_point``, ``seq_len``, ``model_batch_size``,
                ``num_batches_in_buffer``, ``device``, ``layer``,
                ``act_size``, ``batch_size``.
                ``dataset_path`` is required unless ``dataset`` is supplied.
            dataset: Optional pre-built dataset to use instead of loading from
                ``cfg["dataset_path"]``.  Accepts any iterable of dicts with a
                ``"tokens"``, ``"input_ids"``, or ``"text"`` column — including
                a HuggingFace ``Dataset`` / ``IterableDataset`` or a plain
                Python list of dicts.  When supplied, ``dataset_path`` in cfg
                is ignored and dataset restarts are disabled.
        """
        self.model = model
        self._dataset_from_arg = dataset is not None

        if dataset is not None:
            self.dataset = iter(dataset)
        else:
            dataset_path = cfg["dataset_path"]
            dataset_name = cfg.get("dataset_name", None)
            self.dataset = self._load_dataset(dataset_path, dataset_name)

        self.hook_point          = cfg["hook_point"]
        self.context_size        = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size    = cfg["model_batch_size"]
        self.device              = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.cfg                 = cfg
        self.tokenizer           = model.tokenizer
        self.documents_processed = 0

        skip_documents = cfg.get("skip_documents", 0)
        if skip_documents > 0:
            print(f"   Skipping {skip_documents:,} documents...", flush=True)
            log_every = max(10000, skip_documents // 20)
            for i in range(skip_documents):
                try:
                    next(self.dataset)
                except StopIteration:
                    print(f"   Dataset exhausted after skipping {i:,} documents", flush=True)
                    break
                if (i + 1) % log_every == 0:
                    print(f"   Skipped {i + 1:,} / {skip_documents:,} documents...", flush=True)
            print(f"   Done skipping {skip_documents:,} documents", flush=True)
            self.documents_processed = skip_documents

        self.tokens_column = self._get_tokens_column()

    def _load_dataset(self, dataset_path, dataset_name):
        """Load a dataset from a HuggingFace Hub ID or a local path.

        Local path handling:
          - A directory containing Arrow/Parquet shards (saved with
            ``dataset.save_to_disk()``) is loaded with
            ``datasets.load_from_disk()``.
          - A ``.jsonl`` / ``.json`` file is loaded with the HuggingFace
            ``"json"`` reader (each line must be ``{"text": "..."}`` or
            ``{"tokens": [...]}``.
          - A ``.txt`` file is loaded with the HuggingFace ``"text"`` reader
            (one document per line, yielding ``{"text": "..."}``.
          - Any other local path is attempted with ``load_dataset(path)``
            (e.g. a local directory containing a ``dataset_info.json``).

        For HuggingFace Hub datasets use the Hub repo ID as ``dataset_path``
        and optionally pass ``dataset_name`` for a named configuration.
        """
        p = Path(dataset_path)
        if p.exists():
            if p.is_dir():
                from datasets import load_from_disk
                ds = load_from_disk(str(p))
                if hasattr(ds, "to_iterable_dataset"):
                    ds = ds.to_iterable_dataset()
            elif p.suffix in (".jsonl", ".json"):
                ds = load_dataset("json", data_files=str(p), split="train", streaming=True)
            elif p.suffix == ".txt":
                ds = load_dataset("text", data_files=str(p), split="train", streaming=True)
            else:
                ds = load_dataset(str(p), split="train", streaming=True)
        elif dataset_path in [
            "wikitext-2-raw-v1", "wikitext-2-v1",
            "wikitext-103-raw-v1", "wikitext-103-v1",
        ]:
            ds = load_dataset("wikitext", dataset_path, split="train", streaming=True)
        elif dataset_name is not None:
            ds = load_dataset(dataset_path, name=dataset_name, split="train", streaming=True)
        else:
            ds = load_dataset(dataset_path, split="train", streaming=True)
        return iter(ds)

    @classmethod
    def from_dataset(cls, model: HookedRootModule, cfg: dict, dataset) -> "ActivationsStore":
        """Create an ActivationsStore from a pre-built dataset object.

        Use this when your data is already in memory or in a format that
        doesn't fit the ``dataset_path`` convention.

        Args:
            model: A TransformerLens HookedTransformer.
            cfg: Configuration dict (same as ``__init__``).  ``dataset_path``
                may be omitted or set to any placeholder string.
            dataset: Any iterable of dicts with a ``"tokens"``, ``"input_ids"``,
                or ``"text"`` key.  A HuggingFace ``Dataset``, an
                ``IterableDataset``, or a plain Python list all work.

        Returns:
            An ``ActivationsStore`` backed by the supplied dataset.

        Example — use a local list of tokenized sequences::

            import datasets
            rows = [{"tokens": ids.tolist()} for ids in my_token_tensor]
            hf_ds = datasets.Dataset.from_list(rows)
            store = ActivationsStore.from_dataset(model, cfg, hf_ds)

        Example — use a local text file::

            import datasets
            hf_ds = datasets.load_dataset("text",
                        data_files="my_corpus.txt", split="train")
            store = ActivationsStore.from_dataset(model, cfg, hf_ds)
        """
        if "dataset_path" not in cfg:
            cfg = {**cfg, "dataset_path": "__from_dataset__"}
        return cls(model, cfg, dataset=dataset)

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self):
        all_tokens  = []
        target_len  = self.model_batch_size * self.context_size
        while len(all_tokens) < target_len:
            try:
                batch = next(self.dataset)
            except StopIteration:
                if self._dataset_from_arg:
                    raise StopIteration(
                        "ActivationsStore dataset exhausted. "
                        "The dataset passed to from_dataset() does not support restart. "
                        "Use a larger dataset or increase num_tokens."
                    )
                print("   Dataset exhausted, restarting...")
                dataset_path = self.cfg["dataset_path"]
                dataset_name = self.cfg.get("dataset_name", None)
                self.dataset = self._load_dataset(dataset_path, dataset_name)
                batch = next(self.dataset)

            self.documents_processed += 1

            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"], truncate=True, move_to_device=True, prepend_bos=True
                ).squeeze(0)
                all_tokens.extend(tokens.tolist())
            else:
                tokens = batch[self.tokens_column]
                if isinstance(tokens, torch.Tensor):
                    all_tokens.extend(tokens.tolist())
                else:
                    all_tokens.extend(tokens)

        token_tensor = torch.tensor(all_tokens[:target_len], dtype=torch.long, device=self.device)
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg["layer"] + 1,
            )
            result = cache[self.hook_point].float()
            del cache
            return result

    def _fill_buffer(self):
        all_activations = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations  = self.get_activations(batch_tokens).reshape(-1, self.cfg["act_size"])
            all_activations.append(activations)
            del batch_tokens, activations
        result = torch.cat(all_activations, dim=0)
        del all_activations
        return result

    def _get_dataloader(self):
        return DataLoader(
            TensorDataset(self.activation_buffer),
            batch_size=self.cfg["batch_size"],
            shuffle=True,
        )

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            if hasattr(self, "dataloader_iter"):
                del self.dataloader_iter
            if hasattr(self, "dataloader"):
                if hasattr(self.dataloader, "dataset") and hasattr(
                    self.dataloader.dataset, "tensors"
                ):
                    self.dataloader.dataset.tensors = ()
                del self.dataloader
            if hasattr(self, "activation_buffer"):
                del self.activation_buffer

            self.activation_buffer = self._fill_buffer()
            self.dataloader        = self._get_dataloader()
            self.dataloader_iter   = iter(self.dataloader)
            return next(self.dataloader_iter)[0]
