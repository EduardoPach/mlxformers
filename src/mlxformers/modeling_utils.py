from mlx import nn

from mlxformers.configuration_utils import PretrainedConfig
from mlxformers.utils import PushToHubMixin, logging


logger = logging.get_logger(__name__)

class MlxPreTrainedModel(nn.Module, PushToHubMixin):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module) -> None:
        ...

    def init_weights(self) -> None:
        ...

    def post_init(self) -> None:
        ...

    def save_pretrained(self) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls):
        ...
