from typing import Tuple

import torch
from torch import Tensor

from ._C import lltm_backward, lltm_forward  # type: ignore

__all__ = ["lltm", "reference_lltm"]


def lltm(
    input: Tensor, weights: Tensor, bias: Tensor, old_h: Tensor, old_cell: Tensor
) -> Tuple[Tensor, Tensor]:
    return LLTMFunction.apply(input, weights, bias, old_h, old_cell)


class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_forward(
            input, weights, bias, old_h, old_cell
        )
        new_h, new_cell = outputs[:2]
        variables = list(outputs[1:]) + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_h, grad_cell):
        (
            d_old_h,
            d_input,
            d_weights,
            d_bias,
            d_old_cell,
        ) = lltm_backward(
            grad_h, grad_cell, *ctx.saved_tensors
        )
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


def reference_lltm(
    input: Tensor, weights: Tensor, bias: Tensor, old_h: Tensor, old_cell: Tensor
) -> Tuple[Tensor, Tensor]:
    X = torch.cat([old_h, input], dim=1)

    # Compute the input, output and candidate cell gates with one MM.
    gate_weights = torch.nn.functional.linear(X, weights, bias)
    # Split the combined gate weight matrix into its components.
    gates = gate_weights.chunk(3, dim=1)

    input_gate = torch.sigmoid(gates[0])
    output_gate = torch.sigmoid(gates[1])
    # Here we use an ELU instead of the usual tanh.
    candidate_cell = torch.nn.functional.elu(gates[2])

    # Compute the new cell state.
    new_cell = old_cell + candidate_cell * input_gate
    # Compute the new hidden state and output.
    new_h = torch.tanh(new_cell) * output_gate

    return new_h, new_cell
