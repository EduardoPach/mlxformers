from PIL import Image
import requests
from transformers import SamProcessor
from transformers.tokenization_utils import BatchEncoding

from mlxformers.models.sam.modeling_mlx_sam import MlxSamModel

model_id = "facebook/sam-vit-base"

model = MlxSamModel.from_pretrained(model_id)
processor = SamProcessor.from_pretrained(model_id)

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]] # 2D localization of a window


inputs = processor(raw_image, input_points=input_points, return_tensors="np")
inputs = dict(inputs)
# input_points is wrong when return_tensors="np"
# gotta add an extra dimension
inputs["input_points"] = inputs["input_points"][None]
inputs = BatchEncoding(data=inputs, tensor_type="mlx")

outputs = model(**inputs)

print(outputs)