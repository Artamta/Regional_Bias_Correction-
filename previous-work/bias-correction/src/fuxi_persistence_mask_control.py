"""Mask-only attribution control for the persistence-augmented FuXi adapter.

The frozen three-arm experiment cannot distinguish recent-rainfall information
from its two almost-always-active availability-mask pathways.  This module adds
one deliberately nonselectable context: the normalized rainfall positions are
exactly zero, while the two issue-safe availability masks are byte-identical to
those used by ``persistence_lag12_45k``.

Only context materialization and the lazy dataset live here.  All preprocessing
continues to come from the immutable :class:`PersistenceContextBundle` built by
the original experiment, so there is no second normalization or lag definition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import fuxi_allseason_ensemble_calibration as base
from fuxi_persistence_context import (
    BASE_CONTEXT_CHANNEL_COUNT,
    LAG_CONTEXT_CHANNEL_NAMES,
    LEAD_COUNT,
    PERSISTENCE_CONTEXT_CHANNEL_COUNT,
    PersistenceContextBundle,
    PersistenceContextError,
)


MASK_ONLY_ARM = "mask_only_45k"
MASK_ONLY_CONTEXT_CHANNEL_COUNT = PERSISTENCE_CONTEXT_CHANNEL_COUNT
MASK_ONLY_ADDED_CHANNEL_NAMES = (
    "lag_1week_log1p_z_forced_zero",
    "lag_2week_log1p_z_forced_zero",
    "lag_1week_available",
    "lag_2week_available",
)


def _validate_case_index(bundle: PersistenceContextBundle, case_index: int) -> int:
    if not isinstance(bundle, PersistenceContextBundle):
        raise PersistenceContextError("bundle must be a PersistenceContextBundle")
    if isinstance(case_index, (bool, np.bool_)) or not isinstance(
        case_index, (int, np.integer)
    ):
        raise PersistenceContextError("case_index must be an integer")
    index = int(case_index)
    if index < 0 or index >= bundle.case_count:
        raise PersistenceContextError("case_index is out of range")
    return index


def _validate_indices(indices: np.ndarray, case_count: int) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise PersistenceContextError(
            "dataset indices must be a non-empty one-dimensional integer array"
        )
    selected = raw.astype(np.int64, copy=False)
    if np.any(selected < 0) or np.any(selected >= case_count):
        raise PersistenceContextError("dataset indices contain an out-of-range value")
    if np.unique(selected).size != selected.size:
        raise PersistenceContextError("dataset indices must be unique")
    if selected.size > 1 and np.any(np.diff(selected) <= 0):
        raise PersistenceContextError("dataset indices must be strictly increasing")
    return selected.copy()


def mask_only_context_for_case(
    bundle: PersistenceContextBundle,
    case_index: int,
) -> np.ndarray:
    """Return ``[lead,11,latitude,longitude]`` with zero rain and true masks."""

    index = _validate_case_index(bundle, case_index)
    base_fields = base.context_for_case(bundle.base_context, index)
    expected_base = (
        LEAD_COUNT,
        BASE_CONTEXT_CHANNEL_COUNT,
        *bundle.spatial_shape,
    )
    if base_fields.shape != expected_base or not np.isfinite(base_fields).all():
        raise PersistenceContextError("base context materialization is invalid")

    support = np.asarray(bundle.base_context.support, dtype=bool)
    available = (
        bundle.lag_available[index, :, None, None] & support[None]
    ).astype(np.float32)
    zero_rain = np.zeros((2, *bundle.spatial_shape), dtype=np.float32)
    extras = np.concatenate((zero_rain, available), axis=0)
    if extras.shape != (len(LAG_CONTEXT_CHANNEL_NAMES), *bundle.spatial_shape):
        raise PersistenceContextError("mask-only added-channel geometry is invalid")
    fields = np.broadcast_to(
        extras[None],
        (LEAD_COUNT, len(LAG_CONTEXT_CHANNEL_NAMES), *bundle.spatial_shape),
    )
    result = np.concatenate((base_fields, fields), axis=1).astype(
        np.float32, copy=False
    )
    expected = (
        LEAD_COUNT,
        MASK_ONLY_CONTEXT_CHANNEL_COUNT,
        *bundle.spatial_shape,
    )
    if result.shape != expected or not np.isfinite(result).all():
        raise PersistenceContextError("mask-only context is invalid")
    if np.count_nonzero(result[:, BASE_CONTEXT_CHANNEL_COUNT : BASE_CONTEXT_CHANNEL_COUNT + 2]):
        raise PersistenceContextError("mask-only rainfall channels are not exact zero")
    if np.count_nonzero(result[:, BASE_CONTEXT_CHANNEL_COUNT :, ~support]):
        raise PersistenceContextError("mask-only added channels escape scoring support")
    return np.ascontiguousarray(result)


class MaskOnlyCaseDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Lazy dataset for the single nonselectable mask-only arm."""

    def __init__(
        self,
        members: np.ndarray,
        truth: np.ndarray,
        context: PersistenceContextBundle,
        indices: Sequence[int] | np.ndarray,
    ) -> None:
        if not isinstance(context, PersistenceContextBundle):
            raise PersistenceContextError("context must be a PersistenceContextBundle")
        member_values = np.asarray(members)
        truth_values = np.asarray(truth)
        expected_field = (LEAD_COUNT, *context.spatial_shape)
        if member_values.ndim != 5 or member_values.shape[0] != context.case_count:
            raise PersistenceContextError(
                "members must have shape [case,member,lead,latitude,longitude]"
            )
        if tuple(member_values.shape[2:]) != expected_field:
            raise PersistenceContextError("member fields do not match mask-only context")
        if truth_values.shape != (context.case_count, *expected_field):
            raise PersistenceContextError(
                "truth must have shape [case,lead,latitude,longitude] aligned with context"
            )
        self.members = members
        self.truth = truth
        self.context = context
        self.indices = _validate_indices(np.asarray(indices), context.case_count)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, np.integer)
        ):
            raise IndexError("dataset item must be an integer")
        position = int(item)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("dataset item is out of range")
        case_index = int(self.indices[position])
        members = np.array(self.members[case_index], dtype=np.float32, copy=True)
        truth = np.array(self.truth[case_index], dtype=np.float32, copy=True)
        context = mask_only_context_for_case(self.context, case_index)
        return (
            torch.from_numpy(members),
            torch.from_numpy(context),
            torch.from_numpy(truth),
        )
