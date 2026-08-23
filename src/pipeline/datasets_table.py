"""Table/Parquet-based dataset classes for solar wind prediction.

Provides dataset classes that load data from a single Parquet table
plus index files, keeping the entire table in memory for zero-IO
per-sample access.
"""

import os
import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from .config_helpers import (
    get_timeseries_config, get_timeseries_input_variables,
    get_timeseries_target_variables, get_timeseries_normalization_config
)
from .normalizer import Normalizer
from .statistics import compute_statistics_table, split_by_class, undersample

logger = logging.getLogger(__name__)


class TableBaseDataset(Dataset):
    """Dataset backed by a single Parquet table + index files.

    Loads entire table into memory (~17MB for 10 years of 30-min data).
    Each __getitem__ call slices from the in-memory array using integer
    offsets from the reference time — zero disk I/O per sample.
    """

    def __init__(self, config):
        """Initialize table dataset.

        Args:
            config: Hydra configuration object with data.timeseries section
        """
        self.config = config
        self.data_root = config.environment.data_root
        ts_cfg = get_timeseries_config(config)

        # Load entire table into memory
        table_path = os.path.join(self.data_root, ts_cfg.table_file)
        df = pd.read_parquet(table_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # Variable lists
        self.input_variables = get_timeseries_input_variables(config)
        self.target_variables = get_timeseries_target_variables(config)
        self.all_variables = list(dict.fromkeys(
            self.input_variables + self.target_variables
        ))

        # Numeric array (contiguous, read-only for multi-worker safety)
        self.array = df[self.all_variables].values.astype(np.float32)
        self.array.flags.writeable = False

        # Datetime → row index mapping
        self.datetimes = df['datetime'].values
        self.dt_to_row = {ts: i for i, ts in enumerate(self.datetimes)}

        # Window offsets (from config)
        self.input_start = ts_cfg.input_start
        self.input_end = ts_cfg.input_end
        self.target_start = ts_cfg.target_start
        self.target_end = ts_cfg.target_end
        self.input_len = self.input_end - self.input_start
        self.target_len = self.target_end - self.target_start

        # Load index files
        self.train_index = self._load_index(
            os.path.join(self.data_root, ts_cfg.train_index))
        val_index_path = os.path.join(self.data_root, ts_cfg.validation_index)
        self.validation_index = self._load_index(val_index_path)

        # Statistics (from training portion of the table)
        # Per-fold CV runs must override stat_file to avoid leaking stats
        # from a different training period across folds.
        stat_filename = getattr(ts_cfg, "stat_file", "table_stats.pkl")
        stat_path = os.path.join(self.data_root, stat_filename)
        train_rows = [self.dt_to_row[dt] for dt, _ in self.train_index]
        self.stat_dict = compute_statistics_table(
            stat_file_path=stat_path,
            data_array=self.array,
            row_indices=train_rows,
            input_start=self.input_start,
            input_end=self.target_end,
            variables=self.all_variables,
            overwrite=False
        )

        # Normalizer
        norm_config = get_timeseries_normalization_config(config)
        self.normalizer = Normalizer(
            stat_dict=self.stat_dict,
            method_config=norm_config
        )

        # Optional training-anchor filter, applied after the statistics block
        # (normalization stats keep describing the full training split;
        # validation/test anchors are never filtered, so metrics stay
        # comparable across sweeps). Two modes on the TARGET window of
        # `variable`, in raw table units:
        #   peak_min: keep anchors with at least one point >= peak_min
        #             (storm-only training)
        #   peak_max: keep anchors with NO point >= peak_max
        #             (quiet-only training) — the exact complement of
        #             peak_min at the same value, so storm + quiet = all.
        tf_cfg = getattr(ts_cfg, 'train_filter', None)
        peak_min = getattr(tf_cfg, 'peak_min', None) if tf_cfg is not None else None
        peak_max = getattr(tf_cfg, 'peak_max', None) if tf_cfg is not None else None
        if peak_min is not None and peak_max is not None:
            raise ValueError("train_filter: set peak_min or peak_max, not both")
        if peak_min is not None or peak_max is not None:
            threshold = float(peak_min if peak_min is not None else peak_max)
            keep_storm = peak_min is not None
            filter_var_idx = self.all_variables.index(tf_cfg.variable)
            n_before = len(self.train_index)
            kept = []
            for dt, label in self.train_index:
                row = self.dt_to_row[dt]
                window = self.array[row + self.target_start:
                                    row + self.target_end, filter_var_idx]
                # NaN compares False: gap-only windows count as "no storm".
                is_storm = bool((window >= threshold).any())
                if is_storm == keep_storm:
                    kept.append((dt, label))
            mode = (f"peak >= {threshold}" if keep_storm
                    else f"no point >= {threshold}")
            if not kept:
                raise ValueError(
                    f"train_filter ({tf_cfg.variable} {mode}) removed all "
                    f"{n_before} training anchors")
            self.train_index = kept
            logger.info(
                f"Train filter: {tf_cfg.variable} target-window {mode} kept "
                f"{len(kept)}/{n_before} anchors")

        # Pre-compute variable index lookups
        self._input_var_idx = [
            self.all_variables.index(v) for v in self.input_variables
        ]
        self._target_var_idx = [
            self.all_variables.index(v) for v in self.target_variables
        ]

        logger.info(f"Table mode: {len(self.array)} rows loaded from {table_path}, "
                    f"input=[T{self.input_start:+d}:T{self.input_end:+d}] "
                    f"({self.input_len} steps), "
                    f"target=[T{self.target_start:+d}:T{self.target_end:+d}] "
                    f"({self.target_len} steps)")

    def _load_index(self, path: str) -> List[Tuple]:
        """Load index CSV → list of (numpy.datetime64, label) tuples.

        Args:
            path: Path to index CSV file (columns: datetime, label)

        Returns:
            List of (datetime64, label) tuples
        """
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        labels = (df['label'].values if 'label' in df.columns
                  else np.zeros(len(df), dtype=int))
        return [(np.datetime64(dt), int(lbl))
                for dt, lbl in zip(df['datetime'].values, labels)]

    def _read_and_process(self, ref_dt) -> Tuple[np.ndarray, np.ndarray]:
        """Slice and normalize input/target from in-memory array.

        Args:
            ref_dt: Reference time (numpy.datetime64)

        Returns:
            Tuple of (input_array, target_array)
        """
        ref_row = self.dt_to_row[ref_dt]

        # Slice from in-memory array
        raw_input = self.array[ref_row + self.input_start:ref_row + self.input_end]
        raw_target = self.array[ref_row + self.target_start:ref_row + self.target_end]

        # Normalize input variables
        input_cols = []
        for j in self._input_var_idx:
            input_cols.append(
                self.normalizer.normalize_omni(raw_input[:, j],
                                               self.all_variables[j]))
        input_array = np.stack(input_cols, axis=-1)

        # Normalize target variables
        target_cols = []
        for j in self._target_var_idx:
            target_cols.append(
                self.normalizer.normalize_omni(raw_target[:, j],
                                               self.all_variables[j]))
        target_array = np.stack(target_cols, axis=-1)

        return input_array.astype(np.float32), target_array.astype(np.float32)

    @staticmethod
    def _dt_to_name(dt) -> str:
        """Convert numpy datetime64 to filename-safe string (YYYYMMDDHHmmSS)."""
        ts = pd.Timestamp(dt)
        return ts.strftime('%Y%m%d%H%M%S')


class TableTrainDataset(TableBaseDataset):
    """Training dataset for table mode with undersampling support."""

    def __init__(self, config):
        super().__init__(config)

        # Build ref_dts (for slicing) and file_list (for undersampling/display)
        self._ref_dts = [dt for dt, _ in self.train_index]
        self.file_list = [
            (self._dt_to_name(dt), label) for dt, label in self.train_index
        ]

        self.enable_undersampling = config.sampling.enable_undersampling
        self.undersampling_mode = getattr(
            config.sampling, 'undersampling_mode', 'static'
        )

        if self.enable_undersampling and self.undersampling_mode == 'static':
            seed = config.experiment.seed
            subsample, num_pos, num_neg = undersample(
                self.file_list,
                num_subsample=config.sampling.num_subsamples,
                subsample_index=config.sampling.subsample_index,
                seed=seed
            )
            self.file_list = subsample
            logger.info(
                f"Table static undersampling: {num_pos} pos, {num_neg} neg, "
                f"fold {config.sampling.subsample_index}/{config.sampling.num_subsamples}"
            )
        elif self.enable_undersampling and self.undersampling_mode == 'dynamic':
            positive, negative = split_by_class(self.file_list)
            logger.info(
                f"Table dynamic undersampling: {len(positive)} pos, "
                f"{len(negative)} neg (sampler balances each epoch)"
            )

        # Build name→datetime lookup for __getitem__
        self._name_to_dt = {
            self._dt_to_name(dt): dt for dt, _ in self.train_index
        }

        self.num = len(self.file_list)
        logger.info(f"TableTrainDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        name, label = self.file_list[idx]
        ref_dt = self._name_to_dt[name]
        input_array, target_array = self._read_and_process(ref_dt)

        # Data augmentation (training only)
        noise_std = getattr(
            self.config.data.timeseries.augmentation, 'gaussian_noise_std', 0.0
        )
        if noise_std > 0:
            input_array = input_array + np.random.normal(
                0, noise_std, input_array.shape
            ).astype(np.float32)

        label_array = np.array([[label]], dtype=np.float32)

        return {
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': name
        }


class TableValidationDataset(TableBaseDataset):
    """Validation dataset for table mode."""

    def __init__(self, config):
        super().__init__(config)
        self._name_to_dt = {
            self._dt_to_name(dt): dt for dt, _ in self.validation_index
        }
        self.file_list = [
            (self._dt_to_name(dt), label) for dt, label in self.validation_index
        ]
        self.num = len(self.file_list)
        logger.info(f"TableValidationDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        name, label = self.file_list[idx]
        ref_dt = self._name_to_dt[name]
        input_array, target_array = self._read_and_process(ref_dt)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': name
        }


class TableTestDataset(TableBaseDataset):
    """Test dataset for table mode."""

    def __init__(self, config):
        super().__init__(config)
        ts_cfg = get_timeseries_config(config)

        test_index_path = os.path.join(self.data_root, ts_cfg.test_index)
        if os.path.exists(test_index_path):
            test_index = self._load_index(test_index_path)
        else:
            logger.warning(f"Test index not found at {test_index_path}, "
                          f"using validation index")
            test_index = self.validation_index

        self._name_to_dt = {
            self._dt_to_name(dt): dt for dt, _ in test_index
        }
        self.file_list = [
            (self._dt_to_name(dt), label) for dt, label in test_index
        ]
        self.num = len(self.file_list)
        logger.info(f"TableTestDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        name, label = self.file_list[idx]
        ref_dt = self._name_to_dt[name]
        input_array, target_array = self._read_and_process(ref_dt)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': name
        }
