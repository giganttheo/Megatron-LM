# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch

from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.utils import get_pg_rank, get_pg_size

from .utils import VocabUtility


class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel
    ranks. This implementation is used in both fused and unfused cross entropy implementations
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates logits_max."""

        vocab_parallel_logits = vocab_parallel_logits.float()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates predicted logits."""

        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor, predicted_logits: torch.Tensor, sum_exp_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates cross entropy loss."""

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare gradient calculation operands."""

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: torch.Tensor,
        arange_1d: torch.Tensor,
        masked_target_1d: torch.Tensor,
        softmax_update: torch.Tensor,
        grad_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates gradients."""

        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0, tp_group=None):
        """Vocab parallel cross entropy forward function."""

        if tp_group is None:
            tp_group = get_tensor_model_parallel_group()

        vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            vocab_parallel_logits
        )
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_pg_rank(tp_group)
        world_size = get_pg_size(tp_group)
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
            )
        )

        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            r"""
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """  # pylint: disable=line-too-long
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities.
            # So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Vocab parallel cross entropy backward function."""

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        (grad_2d, arange_1d, softmax_update, grad_input) = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
        )

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad

            # Finally elementwise multiplication with the output gradients.
            grad_input.mul_(grad_output.unsqueeze(dim=-1))
        else:
            grad_input = VocabParallelCrossEntropy.calculate_gradients(
                grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
            )

        return grad_input, None, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
            dimension is [sequence_length, batch_size, vocab_size/num_parallel_ranks]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        label_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)

        tp_group: the tensor parallel group over which to all reduce
    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, label_smoothing, tp_group
    )


class _VocabParallelCrossEntropyMultiTarget(torch.autograd.Function):
    """Cross entropy for S independent target sets sharing the SAME logits.

    Used by Token Superposition Training: one set of logits predicts S
    different next-bag tokens, and the loss is the mean of S independent
    cross-entropy terms. The naive approach calls vocab_parallel_cross_entropy
    S times, which recomputes the full O(seq*bs*vocab) softmax normalization
    (max, exp, sum, and their all-reduces) S times even though it is
    IDENTICAL across all S calls -- only the gather-at-target step differs.

    This function computes the shared softmax normalization ONCE, then does
    S cheap gather + tiny-all-reduce passes (O(seq*bs), independent of vocab
    size) instead of S full vocab-wide passes. For vocab ~50k and S in the
    6-16 range (typical TST configs), this removes the dominant remaining
    per-iteration cost after the embedding/schedule fixes.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, targets, tp_group=None):
        """
        Args:
            vocab_parallel_logits: [seq, bs, vocab_shard]
            targets: [S, seq, bs] -- S separate target sets sharing the same logits
            tp_group: tensor-parallel process group

        Returns:
            loss: [seq, bs] -- mean over S of per-target cross-entropy loss
        """
        if tp_group is None:
            tp_group = get_tensor_model_parallel_group()

        S = targets.shape[0]
        assert S >= 1, "targets must have at least one target set along dim 0"

        # --- Shared softmax normalization, computed ONCE ---
        vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            vocab_parallel_logits
        )
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # shifted logits, kept until all S gathers are done (see below), then
        # exponentiated in place once no longer needed in pre-exp form.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        shifted_logits = vocab_parallel_logits

        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = shifted_logits.size()[-1]
        rank = get_pg_rank(tp_group)
        world_size = get_pg_size(tp_group)
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        logits_2d = shifted_logits.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)

        seq, bs = targets.shape[1], targets.shape[2]

        target_masks = torch.empty(
            (S,) + targets.shape[1:], dtype=torch.bool, device=shifted_logits.device
        )
        masked_targets_1d = torch.empty(
            (S, arange_1d.numel()), dtype=torch.long, device=shifted_logits.device
        )
        predicted_logits_all = torch.empty(
            (S, seq, bs), dtype=torch.float32, device=shifted_logits.device
        )

        # Gather all S predicted (shifted) logits FIRST, while shifted_logits
        # still holds pre-exp values -- O(S*seq*bs) gathers, NOT O(S*seq*bs*vocab).
        for i in range(S):
            target_i = targets[i]
            target_mask = (target_i < vocab_start_index) | (target_i >= vocab_end_index)
            masked_target = target_i.clone() - vocab_start_index
            masked_target[target_mask] = 0
            masked_target_1d = masked_target.view(-1)

            predicted_1d = logits_2d[arange_1d, masked_target_1d].clone().contiguous()
            predicted = predicted_1d.view_as(target_i)
            predicted[target_mask] = 0.0

            predicted_logits_all[i] = predicted
            target_masks[i] = target_mask
            masked_targets_1d[i] = masked_target_1d

        # Single all-reduce covering all S predicted-logit tensors at once
        # (stacked), instead of S separate all-reduces -- fewer, larger
        # collectives are cheaper than many tiny ones.
        torch.distributed.all_reduce(
            predicted_logits_all, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )

        # NOW exponentiate in place (shifted_logits no longer needed in
        # pre-exp form) and compute the shared sum_exp / log_sum_exp ONCE.
        exp_logits = shifted_logits
        torch.exp(shifted_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )
        log_sum_exp_logits = torch.log(sum_exp_logits)  # [seq, bs], shared across S

        total_loss = (log_sum_exp_logits.unsqueeze(0) - predicted_logits_all).sum(dim=0)
        mean_loss = total_loss / S

        # softmax = exp_logits / sum_exp_logits, used identically for all S in backward.
        softmax = exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        ctx.save_for_backward(softmax, target_masks, masked_targets_1d)
        ctx.S = S
        ctx.partition_vocab_size = partition_vocab_size

        return mean_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        dL/dlogits = softmax - (1/S) * sum_i onehot(target_i, masked)

        softmax is shared; we start from one copy of it and subtract 1/S at
        each of the S (possibly overlapping) target positions -- S cheap
        scatter ops on top of a single softmax tensor, no repeated exp/sum.
        """
        softmax, target_masks, masked_targets_1d = ctx.saved_tensors
        S = ctx.S
        partition_vocab_size = ctx.partition_vocab_size

        grad_input = softmax  # reuse buffer, softmax not needed after this
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        inv_S = 1.0 / S
        for i in range(S):
            softmax_update = (1.0 - target_masks[i].view(-1).float()) * inv_S
            grad_2d[arange_1d, masked_targets_1d[i]] -= softmax_update

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_parallel_cross_entropy_multi_target(
    vocab_parallel_logits: torch.Tensor,
    targets: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Cross entropy loss for S independent target sets sharing the same logits,
    e.g. Token Superposition Training where one compressed position predicts
    S next-bag tokens. Computes the shared softmax normalization once instead
    of S times -- see _VocabParallelCrossEntropyMultiTarget for details.

    Args:
        vocab_parallel_logits: [sequence_length, batch_size, vocab_size/num_parallel_ranks]
        targets: [S, sequence_length, batch_size] -- S target sets
        tp_group: the tensor parallel group over which to all reduce

    Returns:
        loss: [sequence_length, batch_size] -- mean over S of per-target CE loss
    """
    return _VocabParallelCrossEntropyMultiTarget.apply(vocab_parallel_logits, targets, tp_group)
