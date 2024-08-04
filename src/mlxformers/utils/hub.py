import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from pathlib import Path


import mlx.core as mx
from huggingface_hub import HfApi, RepoUrl, ModelCard, ModelCardData, CommitOperationAdd, CommitInfo
from huggingface_hub.utils import EntryNotFoundError

from .generic import working_or_temp_dir
from .logging import get_logger

def create_and_tag_model_card(
    repo_id: str,
    tags: Optional[List[str]] = None,
    token: Optional[str] = None,
    ignore_metadata_errors: bool = False,
):
    """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`List[str]`, *optional*):
            The list of tags to add in the model card
        token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
    """
    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors)
    except EntryNotFoundError:
        # Otherwise create a simple model card from template
        model_description = "This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated."
        card_data = ModelCardData(tags=[] if tags is None else tags, library_name="transformers")
        model_card = ModelCard.from_template(card_data, model_description=model_description)

    if tags is not None:
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    return model_card

class PushToHubMixin(ABC):
    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}
    
    def _upload_files(
        self, 
        api: HfApi, 
        repo_id: str, 
        work_dir: str, 
        files_timestamps: Dict[str, float], 
        create_pr: bool, 
        commit_message: Optional[str] = None,
        revision: str = None,
        commit_description: str = None,
    ) -> CommitInfo:
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"

        modified_files = [
            f
            for f in os.listdir(work_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(work_dir, f)) > files_timestamps[f]
        ]

        # filter for actual files + folders at the root level
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(work_dir, f)) or os.path.isdir(os.path.join(work_dir, f))
        ]

        operations = []
        # upload standalone files
        for file in modified_files:
            if os.path.isdir(os.path.join(work_dir, file)):
                # go over individual files of folder
                for f in os.listdir(os.path.join(work_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(work_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else:
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(work_dir, file), path_in_repo=file)
                )

        return api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            create_pr=create_pr,
            commit_description=commit_description,
            revision=revision
        )

    @abstractmethod
    def save_pretrained(self) -> None:
        pass
    
    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: Optional[str] = None,
        commit_description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> CommitInfo:
        api = HfApi(token=token)

        repo_url: RepoUrl = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True
        )

        model_card = create_and_tag_model_card(repo_id, tags=tags, token=token)
        
        working_dir = repo_url.repo_id.split("/")[-1]

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

            # Update model card if needed:
            model_card.save(os.path.join(work_dir, "README.md"))


            return self._upload_files(
                api,
                repo_id,
                work_dir,
                files_timestamps,
                create_pr=create_pr,
                commit_message=commit_message,
                revision=revision,
                commit_description=commit_description,
            )
