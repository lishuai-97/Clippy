# This is a modified PyTorch version of the original ClippyAdagrad optimizer from
# https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/experimental/optimizers/clippy_adagrad.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ClippyAdagrad optimizer implementation in PyTorch."""
import torch
from torch.optim import Optimizer

from typing import List, Tuple

def shrink_by_references(tensor: torch.Tensor,
                         references: List[torch.Tensor],
                         relative_factors: List[float],
                         absolute_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shrink a tensor such that it is element-wise smaller than a reference.
    
    Scales the given tensor such that for all index i
    |tensor_i| * scale <=
        sum_j |reference_i| * relative_factors_j + absolute_factor,
    where scale is the maximal scalar such that 0 < scale <= 1.

    Args:
        tensor: A Tensor to shrink.
        references: A sequence of Tensors in a shape broadcastable to `tensor`.
            Provides reference values for each coordinate in `tensor` for the shrinking calculation.
        relative_factors: A sequence of non-negative floats, used with absolute factor to obtain
            the per-element shrinking values.
        absolute_factor: A non-negative float, used with relative factors to obtain the 
            per-element shrinking values.

    Returns:
        A tuple containing the scaled tensor (a tensor of the same shape as the given tensor) 
        and a scalar scaling factor in [0, 1]. When absolute_factor is positive, the scaling 
        factor will also be guaranteed to be positive.

    Raises:
        ValueError: if one of relative_factors is negative, absolute_factor is non-positive 
            or the lengths of references and relative_factors lists are not equal.
    """

    if any(relative_factor < 0 for relative_factor in relative_factors):
        raise ValueError("relative_factors must all be non-negative.")
    if absolute_factor < 0:
        raise ValueError("absolute_factor must be non-negative.")
    if len(references) != len(relative_factors):
        raise ValueError(
            "references and relative_factors must have the same length.  "
            f"Instead they are {len(references)} and {len(relative_factors)}.")

    max_delta = sum(
        (torch.abs(reference) * relative_factor 
         for reference, relative_factor in zip(references, relative_factors)),
         start=absolute_factor)

    # print(f'max_delta type: {type(max_delta)}')
    # if the references or the relative factors are null, convert the max_delta to a tensor
    if not references or not relative_factors:
        max_delta = torch.tensor(max_delta)

    # We are looking for the largest constant 0 <= scale <= 1 such that
    # scale * tensor[i].abs() <= max_delta[i], for all i. Note that both
    # tensor[i] and max_delta[i] may be zeros. If max_delta is zero, then scale
    # must be zero, and if tensor is zero, scale is arbitrary.
    per_element_scale = torch.where(
        tensor == 0., torch.tensor(1.0), max_delta.div(tensor.abs().add_(1e-10)))
    scale = per_element_scale.min().clamp_(max=1.)
    
    return tensor * scale, scale


class ClippyAdagrad(Optimizer):
    r"""An Adagrad variant with adaptive clipping.

    The adaptive clipping mechanism multiplies the learning rate for each model
    parameter w by a factor in (0, 1] that ensures that at each iteration w is never
    changed by more than:
        |w| * variable_relative_threshold
            + accumulator_relative_threshold / sqrt(accum) + absolute_threshold,
    where `accum` is the respective Adagrad accumulator.
    
    Reference: https://arxiv.org/pdf/2302.09178.pdf.
    
    Attributes:
        iterations: The number of training steps this optimizer has run.
        learning_rate: The learning rate constant or schedule.
        clipping_factors: When the argument `export_clipping_factors` is set the True
            will contain a list of the scaling factors used to clip each variable in
            the model. Otherwise, contains an empty list.
    """
    
    def __init__(
            self,
            params,
            lr = 0.001,
            initial_accumulator_value: float = 0.1,
            params_relative_threshold: float = 0.1,
            accumulator_relative_threshold: float = 0.0,
            absolute_threshold: float = 1e-7,
            epsilon: float = 1e-7,
            export_clipping_factors: bool = False,
            clip_accumulator_update: bool = False,
            use_standard_accumulator_update: bool = False,
        ):

        """Initializes the ClippyAdagrad optimizer.
        
        Args:
            lr: Initial value for the learning rate: either a floating positive value, 
                or a schedule. Default to 0.001. Note that `Adagrad` tends to benefit 
                form higher initial learning rate values compared to other optimizers.
                To match the exact form in the original paper, use 1.0.
            initial_accumulator_value: A non-negative floating point value.
                Starting value for the Adagrad accumulator.
            params_relative_threshold: A non-negative floating point value. The
                relative threshold factor for the adaptive clipping, relatively to the
                updated parameters.
            accumulator_relative_threshold: A non-negative floating point value. The
                clipping threshold factor relatively to the inverse square root of the
                Adagrad accumulators. Default to 0.0 but a non-negative value
                (e.g., 1e-3) allows tightening the clipping threshold in later training.
            absolute_threshold: A non-negative floating point value. The absolute 
                clipping threshold constant.
            epsilon: Small floating point value used to maintain numerical stability.
            export_clipping_factors: When set to True, will add an attribute to the
                optimizer, called `clipping_factors`, a list containing the scaling
                factors used to clip each variable in the model. Otherwise,
                the `clipping_factors` attribute is an empty list.
            clip_accumulator_update: When set to True, will also apply clipping on the
                Adagrad accumulator update. This may help improve convergence speed in
                cases where the gradient contains outliers. This cannot be set to True
                when use_standard_accumulator_update is set to True.
            use_standard_accumulator_update: When set to True, will update the
                accumulator before calculating the Adagrad step, as in the classical
                Adagrad method. This cannot be set to True when clip_accumulator_update
                is set to True.
        
        Raises:
            ValueError: If both `clip_accumulator_update` and 
                `use_standard_accumulator_update` are set to True.
        """
        
        if clip_accumulator_update and use_standard_accumulator_update:
            raise ValueError(
                "clip_accumulator_update and use_standard_accumulator_update cannot"
                "both be set to True.")

        defaults = dict(
            lr=lr,
            initial_accumulator_value=initial_accumulator_value,
            params_relative_threshold=params_relative_threshold,
            accumulator_relative_threshold=accumulator_relative_threshold,
            absolute_threshold=absolute_threshold,
            epsilon=epsilon,
            export_clipping_factors=export_clipping_factors,
            clip_accumulator_update=clip_accumulator_update,
            use_standard_accumulator_update=use_standard_accumulator_update,
            )

        super(ClippyAdagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['accumulator'] = torch.full_like(p.data,
                                                           group['initial_accumulator_value'])
                if group['export_clipping_factors']:
                    state['clipping_factor'] = torch.tensor(0.)

                state['step'] += 1
                lr = group['lr']
                epsilon = group['epsilon']
                accumulator = state['accumulator']

                if group['use_standard_accumulator_update']:
                    accumulator.add_(torch.square(grad))
                
                # Note that unlike the standard Adagrad implementation, ClippyAdagrad
                # supports using accumulator value _before_ adding the current gradient to
                # it (by setting `use_standard_accumulator_update=False`). This allows us to
                # update the accumulator using the clipped gradient value, which is not
                # currently known. Also, mathematically, this makes accumulator independent
                # of the current step, which is ususally considered better practice.
                
                # G_t = G_{t-1} + g_t^2
                # r_t = g_t * G_t^{-1/2}
                # delta = lr * r_t = lr * g_t * G_t^{1/2} 
                precondition = torch.rsqrt(accumulator + epsilon)
                delta = lr * grad * precondition

                clipped_delta, clipping_factor = shrink_by_references(
                    delta,
                    references=[p.data, precondition],
                    relative_factors=[
                        group['params_relative_threshold'],
                        group['accumulator_relative_threshold'],
                    ],
                    absolute_factor=group['absolute_threshold'],
                )

                if group['export_clipping_factors']:
                    state['clipping_factor'] = clipping_factor
                
                if not group['use_standard_accumulator_update']:
                    # Delayed accumulator update, This allows clipping accumulator update.
                    if group['clip_accumulator_update']:
                        # Clip the accumulator update: this act like clipping the gradient
                        # before adding it to the optimizer. This is a good option when the
                        # gradient is an outlier.
                        accumulator_update = grad * clipping_factor
                    else:
                        # Does not clip the accumulator update: This a good option in cases
                        # where the gradient increases during training, and allows for quicker
                        # adjustment to the increase by the accumulator.
                        accumulator_update = grad
                    
                    accumulator.add_(torch.square(accumulator_update))

                p.data.sub_(clipped_delta)

        return loss