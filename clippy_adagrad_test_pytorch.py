# This is a modified PyTorch version of the original ClippyAdagrad optimizer test script from
# https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/experimental/optimizers/clippy_adagrad_test.py
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

"""Tests for ClippyAdagrad in PyTorch."""

import torch
import unittest

import clippy_adagrad_pytorch as clippy_adagrad

class ClipByReferenceTest(unittest.TestCase):

    # This method is used to compare whether torch.Tensor objects are almost equal
    def assertAllCloseAccordingToType(self, a, b, rtol=1e-4, atol=1e-7):
        self.assertTrue(torch.allclose(a, b, rtol=rtol, atol=atol))

    # The following are test cases using the shrink_by_references function
    def test_scalar_clip(self):
        tensor = torch.tensor(2.0)
        references = [torch.tensor(4.0)]
        relative_factors = [0.1]
        absolute_factor = 0.02
        expected_clipped = torch.tensor(0.42)
        expected_scale = torch.tensor(0.21)

        clipped, scale = clippy_adagrad.shrink_by_references(tensor, 
                                                             references, 
                                                             relative_factors, 
                                                             absolute_factor)
        
        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)


    def test_scalar_multiple_clip(self):
        tensor = torch.tensor(2.0)
        references = [torch.tensor(4.0), torch.tensor(-5.0)]
        relative_factors = [0.1, 0.2]
        absolute_factor = 0.02
        expected_clipped = torch.tensor(4 * .1 + 5 * .2 + 0.02)
        expected_scale = torch.tensor((4 * .1 + 5 * .2 + .02) / 2)

        clipped, scale = clippy_adagrad.shrink_by_references(tensor,
                                                             references,
                                                             relative_factors,
                                                             absolute_factor)

        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)

    
    def test_scalar_empty_reference(self):
        tensor = torch.tensor(2.0)
        references = []
        relative_factors = []
        absolute_factor = 0.02
        expected_clipped = torch.tensor(0.02)
        expected_scale = torch.tensor(0.01)
    
        clipped, scale = clippy_adagrad.shrink_by_references(tensor,
                                                             references,
                                                             relative_factors,
                                                             absolute_factor)
        
        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)

    def test_scalar_empty_reference_1(self):
        tensor = torch.tensor(0.)
        references = []
        relative_factors = []
        absolute_factor = 0.
        expected_clipped = torch.tensor(0.)
        expected_scale = torch.tensor(1.)

        clipped, scale = clippy_adagrad.shrink_by_references(tensor,
                                                             references,
                                                             relative_factors,
                                                             absolute_factor)
        
        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)

    def test_tensor_clip(self):
        tensor = torch.tensor([1., 1.])
        references = [torch.tensor([1., 0.1])]
        relative_factors = [0.1]
        absolute_factor = 0.01
        expected_clipped = torch.tensor([0.02, 0.02])
        expected_scale = torch.tensor(0.02)

        clipped, scale = clippy_adagrad.shrink_by_references(tensor,
                                                             references,
                                                             relative_factors,
                                                             absolute_factor)
        
        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)

    def test_tensor_clip_zero_absolute_factor(self):
        tensor = torch.tensor([1., 1., 0., 0.])
        references = [torch.tensor([1., 0.1, 1., 0.])]
        relative_factors = [0.1]
        absolute_factor = 0.
        expected_clipped = torch.tensor([0.01, 0.01, 0., 0.])
        expected_scale = torch.tensor(0.01)

        clipped, scale = clippy_adagrad.shrink_by_references(tensor,
                                                             references,
                                                             relative_factors,
                                                             absolute_factor)
        
        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)
        
    def test_tensor_clip_zero_reference(self):
        tensor = torch.tensor([1., 1., 0., 0.])
        references = [torch.tensor([1., 0., 1., 0.])]
        relative_factors = [0.1]
        absolute_factor = 0.
        expected_clipped = torch.tensor([0., 0., 0., 0.])
        expected_scale = torch.tensor(0.)

        clipped, scale = clippy_adagrad.shrink_by_references(tensor,
                                                             references,
                                                             relative_factors,
                                                             absolute_factor)
        
        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)

    def test_broadcast(self):
        tensor = torch.tensor([[1., 2.], [1., 2.]])
        references = [torch.tensor(1.)]
        relative_factors = [0.1]
        absolute_factor = 0.1
        expected_clipped = torch.tensor([[0.1, 0.2], [0.1, 0.2]])
        expected_scale = torch.tensor(0.1)

        clipped, scale = clippy_adagrad.shrink_by_references(tensor,
                                                             references,
                                                             relative_factors,
                                                             absolute_factor)
        
        self.assertAllCloseAccordingToType(expected_clipped, clipped)
        self.assertAllCloseAccordingToType(expected_scale, scale)


class ClippyAdagradTest(unittest.TestCase):

    def assertAllCloseAccordingToType(self, a, b, rtol=1e-4, atol=1e-7):
        # This method is used to compare whether torch.Tensor objects are almost equal
        self.assertTrue(torch.allclose(a, b, rtol=rtol, atol=atol))

    def test_single_step_no_clip(self):
        learning_rate = 0.1
        initial_accumulator_sqrt = 0.1
        optimizer = clippy_adagrad.ClippyAdagrad(
            [torch.tensor([1.0, 2.0], dtype=torch.float32)],
            lr=learning_rate,
            initial_accumulator_value=initial_accumulator_sqrt**2,
            export_clipping_factors=True)
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
        g = torch.tensor([0.1, 0.15])

        # we simulate sparse update by directly manipulating the variable in PyTorch
        sparse_x = torch.tensor([[3.0, 4.0], [1.0, 2.0]], requires_grad=True)
        sparse_g = torch.tensor([[0, 0], [0.1, 0.15]])

        # Execute optimiser step
        optimizer.zero_grad()
        x.grad = g
        sparse_x.grad = sparse_g
        optimizer.step()

        # self.assertAllCloseAccordingToType(x, torch.tensor([
        #     1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
        #     2.0 - learning_rate * 0.15 / initial_accumulator_sqrt
        # ]))
        # self.assertAllCloseAccordingToType(
        #     sparse_x, torch.tensor([[3.0, 4.0],
        #                             [
        #                                 1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
        #                                 2.0 - learning_rate * 0.15 / initial_accumulator_sqrt
        #                             ]])
        # )

        # The state in PyTorch is contained in the optimizer's state
        accumulators = optimizer.state[sparse_x]['accumulator']
        self.assertAllCloseAccordingToType(optimizer.state[x]['accumulator'], torch.tensor([
            initial_accumulator_sqrt**2 + 0.1**2,
            initial_accumulator_sqrt**2 + 0.15**2
        ]))
        self.assertAllCloseAccordingToType(
            accumulators,
            torch.tensor([[initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
                          [initial_accumulator_sqrt**2 + 0.1**2, initial_accumulator_sqrt**2 + 0.15**2]])
        )
        if optimizer.export_clipping_factors:
            clipping_factors = optimizer.state[x]['clipping_factor']
            self.assertAllCloseAccordingToType(clipping_factors, torch.tensor([1.0, 1.0]))


if __name__ == '__main__':
    unittest.main()