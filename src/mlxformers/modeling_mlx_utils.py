import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import custom_object_save
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    cached_file,
    copy_func,
    download_url,
    has_file,
    is_remote_url,
    logging,
)
from transformers.utils.hub import convert_file_size_to_int, get_checkpoint_shard_files


MLX_WEIGHTS_NAME = "mlx_model.safetensors"
MLX_WEIGHTS_INDEX_NAME = "mlx_model.safetensors.index.json"

logger = logging.get_logger(__name__)

ACT2FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "relu6": nn.relu6,
    "mish": nn.mish,
    "leaky_relu": nn.leaky_relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "sigmoid": nn.sigmoid,
    "quick_gelu": nn.gelu_fast_approx,
    "gelu_accurate": nn.gelu_approx,
    "tanh": nn.tanh,
}


def is_file(*paths: str) -> bool:
    """
    Joins in order all the paths provided and checks if the resulting path is a file.
    """
    return os.path.isfile(os.path.join(*paths))


# While MLX does not support ConvTranspose2d, we can use the following implementation
class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
    ):
        super().__init__()

        kernel_size, stride, padding = [(x, x) if isinstance(x, int) else x for x in (kernel_size, stride, padding)]
        # Add +1 to padding to match the behavior of PyTorch
        padding = (padding[0] + 1, padding[1] + 1)

        scale = math.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1:2]}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        y = mx.conv_general(
            x,
            self.weight,
            stride=1,
            padding=self.padding,
            kernel_dilation=self.dilation,
            input_dilation=self.stride,
            flip=True,
        )
        if "bias" in self:
            y = y + self.bias
        return y


def local_case(base_path: str, from_pt: bool) -> Tuple[str, bool]:
    if is_file(base_path, MLX_WEIGHTS_NAME):
        archive_file = os.path.join(base_path, MLX_WEIGHTS_NAME)
        is_sharded = False
    elif is_file(base_path, MLX_WEIGHTS_INDEX_NAME):
        archive_file = os.path.join(base_path, MLX_WEIGHTS_INDEX_NAME)
        is_sharded = True
    elif is_file(base_path, SAFE_WEIGHTS_NAME):
        archive_file = os.path.join(base_path, SAFE_WEIGHTS_NAME)
        is_sharded = False
    elif is_file(base_path, SAFE_WEIGHTS_INDEX_NAME):
        archive_file = os.path.join(base_path, SAFE_WEIGHTS_INDEX_NAME)
        is_sharded = True
    elif from_pt and is_file(base_path, WEIGHTS_NAME):
        archive_file = os.path.join(base_path, WEIGHTS_NAME)
        is_sharded = False
        raise NotImplementedError("Support for pytorch checkpoints is coming soon!")
    elif from_pt and is_file(base_path, WEIGHTS_INDEX_NAME):
        archive_file = os.path.join(base_path, WEIGHTS_INDEX_NAME)
        is_sharded = True
        raise NotImplementedError("Support for pytorch checkpoints is coming soon!")
    # At this point we didn't find a file so we need to raise an error
    elif is_file(base_path, WEIGHTS_NAME) or is_file(base_path, WEIGHTS_INDEX_NAME):
        raise EnvironmentError(
            f"Error no file named {MLX_WEIGHTS_NAME} found in directory {base_path} "
            "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
            "weights."
        )
    else:
        raise EnvironmentError(
            f"Error no file named {MLX_WEIGHTS_NAME}, {SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
            f"{base_path}."
        )

    return archive_file, is_sharded


# TODO: this looks ugly
def cached_case(
    pretrained_model_name_or_path: str,
    from_pt: bool,
    cached_file_kwargs: Dict[str, Any],
    has_file_kwargs: Dict[str, Any],
) -> Tuple[str, str, bool]:
    if from_pt:
        filename = WEIGHTS_NAME
        raise NotImplementedError("Support for pytorch checkpoints is coming soon!")
    else:
        filename = MLX_WEIGHTS_NAME

    try:
        is_sharded = False
        # Load from URL or cache if already cached
        resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
        if resolved_archive_file is None and filename == MLX_WEIGHTS_NAME:
            resolved_archive_file = cached_file(
                pretrained_model_name_or_path, MLX_WEIGHTS_INDEX_NAME, **cached_file_kwargs
            )
            if resolved_archive_file is not None:
                is_sharded = True

        # Maybe the checkpoint is pytorch sharded, we try to grab the pytorch index name in this case.
        if resolved_archive_file is None and from_pt:
            resolved_archive_file = cached_file(
                pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs
            )
            if resolved_archive_file is not None:
                is_sharded = True
            raise NotImplementedError("Support for pytorch checkpoints is coming soon!")

        # If we still haven't found anything, look for `safetensors`.
        if resolved_archive_file is None:
            # No support for sharded safetensors yet, so we'll raise an error if that's all we find.
            filename = SAFE_WEIGHTS_NAME
            resolved_archive_file = cached_file(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME, **cached_file_kwargs)

        # Maybe is a sharded checkpoint using safetensors
        if resolved_archive_file is None:
            resolved_archive_file = cached_file(
                pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME, **cached_file_kwargs
            )
            if resolved_archive_file is not None:
                is_sharded = True

        # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
        # result when internet is up, the repo and revision exist, but the file does not.
        if resolved_archive_file is None:
            # Otherwise, maybe there is a TF or Torch model file.  We try those to give a helpful error
            # message.
            if has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                    f" {MLX_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                    " load this model from those weights."
                )
            elif has_file(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **has_file_kwargs):
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                    f" {MLX_WEIGHTS_INDEX_NAME} but there is a sharded file for PyTorch weights. Use"
                    " `from_pt=True` to load this model from those weights."
                )
            else:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                    f" {MLX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                )
    except EnvironmentError:
        # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
        # to the original exception.
        raise
    except Exception:
        # For any other exception, we throw a generic error.
        raise EnvironmentError(
            f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
            " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
            f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
            f" directory containing a file named {MLX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
        )

    return resolved_archive_file, filename, is_sharded


def mlx_shard_checkpoint(
    weights: List[Tuple[str, mx.array]], max_shard_size: str = "10GB"
) -> Tuple[Dict[str, Dict[str, mx.array]], Dict[str, Any]]:
    """
    Shard the checkpoint if it is too big to fit in a single file.

    Args:
        params (`tuple[str, mx.array]`):
            The parameters to be saved.
        max_shard_size (`str`):
            The maximum size of a shard.

    Returns:
        `Dict[str, Dict[str, mx.array]]`:
            A dictionary of sharded parameters.
        `Dict[str, Any]`:
            The index of the sharded parameters.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    # flatten the weights to chunk
    for name, param in weights:
        weight_size = param.size * param.dtype.size  # in bytes

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0

        current_block[name] = param
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {MLX_WEIGHTS_NAME: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = MLX_WEIGHTS_NAME.replace(
            ".safetensors", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for weight_name in shard.keys():
            weight_map[weight_name] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


# TODO: Currently only expecting .safetensors for MLX, but can also accept .npz and .gguf
# TODO: Add GenerationMixin (when this is done should also add generation config to `save_pretrained`)
# TODO: Add support for Pytorch .bin files (from_pt=True)
# TODO: Allow sharded `save_pretrained` for MLX
class MlxPreTrainedModel(nn.Module, PushToHubMixin):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _missing_keys = set()

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module) -> None: ...

    def init_weights(self) -> None: ...

    def post_init(self) -> None: ...

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        params=None,
        push_to_hub=False,
        max_shard_size="10GB",
        token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        if token is not None:
            kwargs["token"] = token

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # get abs dir
        save_directory = os.path.abspath(save_directory)
        # save config as well
        self.config.architectures = [self.__class__.__name__[4:]]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        self.config.save_pretrained(save_directory)

        # save model
        weights_name = MLX_WEIGHTS_NAME
        output_model_file = os.path.join(save_directory, weights_name)

        shards, index = mlx_shard_checkpoint(tree_flatten(self.parameters()), max_shard_size)
        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
            ):
                os.remove(full_filename)

        if index is None:
            mx.save_safetensors(output_model_file, shards[weights_name])

        else:
            save_index_file = os.path.join(save_directory, MLX_WEIGHTS_INDEX_NAME)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
            for shard_file, shard in shards.items():
                output_shard_file = os.path.join(save_directory, shard_file)
                mx.save_safetensors(output_shard_file, shard)

        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    @classmethod
    def load_mlx_sharded_weights(cls, shard_files: List[str]) -> Dict[str, mx.array]:
        """
        Load sharded weights from multiple files. This is a memory efficient loading
        since `mx.load` is lazy.

        Args:
            shard_files (`List[str]`):
                List of paths to the sharded weights files

        Returns:
            `Dict[str, mx.array]`:
                A dictionary of weights
        """
        weights = {}
        for shard_file in shard_files:
            shard = mx.load(shard_file)
            weights.update(shard)
        return weights

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "MlxPreTrainedModel":
        if pretrained_model_name_or_path is None:
            raise ValueError("Please provide a model name or path")

        from_pt = kwargs.pop("from_pt", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        user_agent = {"file_type": "model", "framework": "flax", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                _commit_hash=commit_hash,
                **kwargs,
            )
        else:
            model_kwargs = kwargs.copy()

        if commit_hash is None:
            commit_hash = getattr(config, "_commit_hash", None)

        # TOOO: will check only for dir existence ignoring subfolder

        is_local = os.path.isdir(pretrained_model_name_or_path)
        is_remote = is_remote_url(pretrained_model_name_or_path)
        is_sharded = False

        if is_local:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            base_path = os.path.join(pretrained_model_name_or_path, subfolder)
            archive_file, is_sharded = local_case(base_path, from_pt)
        elif is_remote:
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
        else:
            cached_file_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "local_files_only": local_files_only,
                "token": token,
                "user_agent": user_agent,
                "revision": revision,
                "subfolder": subfolder,
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
                "_commit_hash": commit_hash,
            }
            has_file_kwargs = {
                "revision": revision,
                "proxies": proxies,
                "token": token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }

            resolved_archive_file, filename, is_sharded = cached_case(
                pretrained_model_name_or_path,
                from_pt,
                cached_file_kwargs,
                has_file_kwargs,
            )

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
            filename = resolved_archive_file.split(os.path.sep)[-1]
        else:
            logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            raise NotImplementedError("Support for pytorch checkpoints is coming soon!")
        elif is_sharded:
            weights_dict = cls.load_sharded_weights(resolved_archive_file)
        else:
            weights_dict = mx.load(resolved_archive_file)

        weights = tree_unflatten(list(weights_dict.items()))
        # TODO: Missing warning about non-initialized weights
        model.update(weights)

        if (filename == SAFE_WEIGHTS_INDEX_NAME) or (filename == SAFE_WEIGHTS_NAME) or from_pt:
            logger.info(
                "The checkpoint weights were saved using something other than `MLX`."
                "Convolution layers weights will be reshaped to `channel_last` to match the `MLX` format."
            )
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    logger.debug(f"Reshaping convolution layer {name} to `channel_last`.")
                    # Modification only applied to weight not bias
                    module.apply(lambda param: param.moveaxis(1, -1) if param.ndim > 1 else param)
                if isinstance(module, ConvTranspose2d):
                    logger.debug(f"Reshaping transpose convolution layer {name} to `channel_last`.")
                    module.apply(lambda param: param.moveaxis(0, -1) if param.ndim > 1 else param)

        return model

    @classmethod
    def register_for_auto_class(cls, auto_class="MlxAutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"MlxAutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


# To update the docstring, we need to copy the method, otherwise we change the original docstring.
MlxPreTrainedModel.push_to_hub = copy_func(MlxPreTrainedModel.push_to_hub)
if MlxPreTrainedModel.push_to_hub.__doc__ is not None:
    MlxPreTrainedModel.push_to_hub.__doc__ = MlxPreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="FlaxAutoModel", object_files="model checkpoint"
    )
