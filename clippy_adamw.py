"""ClippyAdamW optimizer implementation in PyTorch."""

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


class ClippyAdamW(Optimizer):
    r"""An AdamW variant with adaptive clipping.
    
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
            lr=0.02,
            weight_decay=0.2,
            betas=(0.9, 0.99),
            eps=1e-6,
            relative_threshold: float = 0.1,
            absolute_threshold: float = 1e-7,
            export_clipping_factors: bool = False,
            precision='amp_bfloat16',
            custom_scalar=65536,
        ):
        
        """Initializes the ClippyAdamW optimizer.
        
        Args:
            lr: Initial value for the learning rate: either a floating positive value, 
                or a schedule. Default to 0.02.
            relative_threshold: A non-negative floating point value. The
                relative threshold factor for the adaptive clipping, relatively to the
                updated parameters.
            absolute_threshold: A non-negative floating point value. The absolute 
                clipping threshold constant.
            epsilon: Small floating point value used to maintain numerical stability.
            export_clipping_factors: When set to True, will add an attribute to the
                optimizer, called `clipping_factors`, a list containing the scaling
                factors used to clip each variable in the model. Otherwise,
                the `clipping_factors` attribute is an empty list.
        
        Raises:
            ValueError: If both `clip_accumulator_update` and 
                `use_standard_accumulator_update` are set to True.
        """
        
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            relative_threshold=relative_threshold,
            absolute_threshold=absolute_threshold,
            export_clipping_factors=export_clipping_factors,
        )

        super(ClippyAdamW, self).__init__(params, defaults)

        self.eps = eps

        # Set precision to "custom_fp16" if you want to use a fixed loss scalar, custom_scalar, which is divided out in the update step.
        # If you do this, call (custom_scalar * loss).backward() instead of loss.backward().
        self.precision = precision
        self.custom_scalar = custom_scalar

        for group in self.param_groups:
            group['step'] = 1.

        print('>>>> Using ClippyAdamW-v1 <<<<')

    def __setstate__(self, state):
        super(ClippyAdamW, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            step = group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                theta = p.data
                param_state = self.state[p]

                if self.precision == 'custom_fp16':
                    g = p.grad.data / self.custom_scalar
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)

                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g, g, value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)
                
                # ClippyAdamW = Adamw + Clippy (https://dl.acm.org/doi/pdf/10.1145/3580305.3599846) applied tensor-wise.
                # updates
                delta = lr * v / denominator

                clipped_delta, clipping_factor = shrink_by_references(
                    delta,
                    references=[theta],
                    relative_factors=[group['relative_threshold']],
                    absolute_factor=group['absolute_threshold'],
                )

                if group['export_clipping_factors']:
                    param_state['clipping_factor'] = clipping_factor

                theta = theta.mul_(1.0 - lr * weight_decay).add_(-clipped_delta)

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
            
            group['step'] += 1



class ClippyAdamWv2(Optimizer):
    r"""An AdamW variant with adaptive clipping.
    
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
            lr=0.02,
            weight_decay=0.2,
            betas=(0.9, 0.99),
            eps=1e-6,
            relative_threshold: float = 0.1,
            absolute_threshold: float = 1e-7,
            export_clipping_factors: bool = False,
            precision='amp_bfloat16',
            custom_scalar=65536,
        ):
        
        """Initializes the ClippyAdamW optimizer.
        
        Args:
            lr: Initial value for the learning rate: either a floating positive value, 
                or a schedule. Default to 0.02.
            relative_threshold: A non-negative floating point value. The
                relative threshold factor for the adaptive clipping, relatively to the
                updated parameters.
            absolute_threshold: A non-negative floating point value. The absolute 
                clipping threshold constant.
            epsilon: Small floating point value used to maintain numerical stability.
            export_clipping_factors: When set to True, will add an attribute to the
                optimizer, called `clipping_factors`, a list containing the scaling
                factors used to clip each variable in the model. Otherwise,
                the `clipping_factors` attribute is an empty list.
        
        Raises:
            ValueError: If both `clip_accumulator_update` and 
                `use_standard_accumulator_update` are set to True.
        """
        
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            relative_threshold=relative_threshold,
            absolute_threshold=absolute_threshold,
            export_clipping_factors=export_clipping_factors,
        )

        super(ClippyAdamWv2, self).__init__(params, defaults)

        self.eps = eps

        # Set precision to "custom_fp16" if you want to use a fixed loss scalar, custom_scalar, which is divided out in the update step.
        # If you do this, call (custom_scalar * loss).backward() instead of loss.backward().
        self.precision = precision
        self.custom_scalar = custom_scalar

        for group in self.param_groups:
            group['step'] = 1.

        print('>>>> Using ClippyAdamW-v2 <<<<')

    def __setstate__(self, state):
        super(ClippyAdamWv2, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            step = group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                theta = p.data
                param_state = self.state[p]

                if self.precision == 'custom_fp16':
                    g = p.grad.data / self.custom_scalar
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)

                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g, g, value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)
                
                # ClippyAdamW = Adamw + Clippy (https://dl.acm.org/doi/pdf/10.1145/3580305.3599846) applied tensor-wise.
                # updates
                delta = v / denominator

                sigma_numerator = group['relative_threshold'] * torch.abs(theta) + group['absolute_threshold']
                sigma_denominator = torch.abs(delta) * lr
                clipping_factor = torch.min(torch.Tensor([1., torch.min(sigma_numerator / sigma_denominator)]))

                if group['export_clipping_factors']:
                    param_state['clipping_factor'] = clipping_factor

                theta = theta.mul_(1.0 - lr * weight_decay).add_(-lr*clipping_factor*delta)

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
            
            group['step'] += 1