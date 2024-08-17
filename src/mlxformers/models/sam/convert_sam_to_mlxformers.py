from transformers import AutoProcessor
from transformers.utils import logging

from mlxformers import MlxSamModel


logger = logging.get_logger(__name__)


def main(from_repo_id: str, to_repo_id) -> None:
    logger.info(f"Loading weights to MlxSamModel from {from_repo_id}...")
    mlx_model = MlxSamModel.from_pretrained(from_repo_id)
    sam_processor = AutoProcessor.from_pretrained(from_repo_id)

    logger.info(f"Saving MlxSamModel to {to_repo_id}...")
    mlx_model.push_to_hub(to_repo_id, tags=["mlx", "mlxformers"])
    sam_processor.push_to_hub(to_repo_id)


if __name__ == "__main__":
    main("facebook/sam-vit-base", "EduardoPacheco/mlx-sam-vit-base")
