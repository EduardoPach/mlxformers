import unittest
from typing import Any, Dict

import numpy as np
import requests
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import cached_property

from mlxformers.models.sam.modeling_mlx_sam import MlxSamModel


class MlxSamModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return SamProcessor.from_pretrained("facebook/sam-vit-base")

    def _load_data(self) -> Dict[str, Any]:
        img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        input_points = [[[450, 600]]]  # 2D localization of a window

        return {"images": raw_image, "input_points": input_points}

    def test_mlx_sam_model_outputs(self) -> None:
        mlx_model = MlxSamModel.from_pretrained("facebook/sam-vit-base")
        pt_model = SamModel.from_pretrained("facebook/sam-vit-base")

        mlx_model.eval()
        pt_model.eval()

        processor = self.default_processor

        data = self._load_data()

        inputs = processor(**data, return_tensors="np")
        inputs = dict(inputs)
        # input_points is wrong when return_tensors="np"
        # gotta add an extra dimension
        inputs["input_points"] = inputs["input_points"][None]
        pt_inputs = BatchEncoding(data=inputs.copy(), tensor_type="pt")
        mlx_inputs = BatchEncoding(data=inputs.copy(), tensor_type="mlx")

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs)

        mlx_outputs = mlx_model(**mlx_inputs)

        pt_pred_masks = pt_outputs.pred_masks
        mlx_pred_masks = mlx_outputs.pred_masks

        # Convert to numpy
        pt_pred_masks = pt_pred_masks.cpu().numpy()
        mlx_pred_masks = np.array(mlx_pred_masks)

        # Assert shapes
        self.assertEqual(pt_pred_masks.shape, mlx_pred_masks.shape)

        # Assert all close
        self.assertTrue(np.allclose(pt_pred_masks, mlx_pred_masks, atol=1e-3))

        # Assert Postprocess masks
        pt_postprocessed_masks = processor.post_process_masks(
            masks=list(pt_pred_masks),
            original_sizes=pt_inputs.original_sizes,
            reshaped_input_sizes=pt_inputs.reshaped_input_sizes,
            binarize=False,
        )

        mlx_postprocessed_masks = mlx_model.post_process_masks(
            masks=list(mlx_pred_masks),
            original_sizes=mlx_inputs.original_sizes,
            reshaped_input_sizes=mlx_inputs.reshaped_input_sizes,
            pad_size=processor.image_processor.pad_size,
            binarize=False,
        )

        for pt_mask, mlx_mask in zip(pt_postprocessed_masks, mlx_postprocessed_masks):
            self.assertTrue(np.allclose(pt_mask, mlx_mask, atol=1e-3))
