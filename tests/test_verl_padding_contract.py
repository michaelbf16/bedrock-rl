"""Executable contract for the Qwen-VL padding patch at the verl pin."""

import unittest
import importlib.util
from unittest.mock import patch


try:
    import torch
    from tensordict import TensorDict
except ImportError:  # The ordinary test environment may lack tensor deps.
    torch = None

HAS_VERL = importlib.util.find_spec("verl") is not None


@unittest.skipIf(torch is None or not HAS_VERL,
                 "torch/tensordict/verl are training dependencies")
class VerlPaddingContractTests(unittest.TestCase):
    def test_four_axis_position_ids_are_ragged_over_sequence(self):
        # If verl is installed but this pinned surface or one of its required
        # imports drifted, the import must fail this contract rather than turn
        # a broken training environment into an optional-dependency skip.
        from verl.workers.utils import padding as verl_padding

        def unpad_input(tensor, mask):
            indices = mask.reshape(-1).nonzero().flatten()
            values = tensor.reshape(-1, tensor.shape[-1])[indices]
            lengths = mask.sum(dim=1, dtype=torch.int32)
            offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
            return values, indices, offsets, int(lengths.max())

        # Avoid making this CPU contract depend on flash-attn. Patch the
        # already-imported padding module directly so test ordering cannot
        # leave its helper references bound before a sys.modules stub.
        position_ids = torch.arange(2 * 4 * 5).reshape(2, 4, 5)
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        data = TensorDict({
            "input_ids": torch.arange(10).reshape(2, 5),
            "attention_mask": mask,
            "response_mask": mask.clone(),
            "position_ids": position_ids,
        }, batch_size=2)

        with (patch.object(verl_padding, "unpad_input", unpad_input),
              patch.object(
                  verl_padding, "index_first_axis",
                  lambda tensor, indices: tensor[indices])):
            packed = verl_padding.left_right_2_no_padding(data)
        nested = packed["position_ids"]
        self.assertTrue(nested.is_nested)
        self.assertEqual(nested._ragged_idx, 2)
        self.assertEqual(nested.offsets().tolist(), [0, 3, 8])
        rows = nested.unbind()
        self.assertEqual([tuple(row.shape) for row in rows], [(4, 3), (4, 5)])
        torch.testing.assert_close(rows[0], position_ids[0, :, 2:])
        torch.testing.assert_close(rows[1], position_ids[1])


if __name__ == "__main__":
    unittest.main()
