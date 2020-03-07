#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
from tensorflow/python/ops/losses/losses_impl.py #log_loss
"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss
LOSSES = "losses"
SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"
def cross_entropy_loss(labels, predictions, weights=1.0, epsilon=1e-7, scope=None, loss_collection=LOSSES, reduction = SUM_BY_NONZERO_WEIGHTS):
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with ops.name_scope(scope, "log_loss",
                        (predictions, labels, weights)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = -math_ops.multiply(
            labels,
            math_ops.log(predictions + epsilon)) - math_ops.multiply(
            (1 - labels), math_ops.log(1 - predictions + epsilon))
        return compute_weighted_loss(
            losses, weights, scope, loss_collection, reduction=reduction)