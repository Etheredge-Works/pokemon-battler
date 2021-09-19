import numpy as np
from PIL import Image
from kedro.io.core import AbstractVersionedDataSet, get_filepath_str, Version
import pytorch_lightning as pl
from typing import Tuple, Union, Dict, Any
import json
from kedro.io.core import get_protocol_and_path
from pathlib import PurePosixPath
import fsspec

class LightningDataSet(AbstractVersionedDataSet):
    MODEL_FILE = "model.ckpt"
    PARAMS_FILE = "params.json"

    def __init__(self, filepath: str, version: Version = None):
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
            version: The version of the dataset being saved and loaded.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )
    def _save(self, data: Tuple[pl.LightningModule, Dict]) -> None:
        """Saves image data to the specified filepath."""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        model = data[0]
        model.save_checkpoint(f"{save_path}/{self.MODEL_FILE}")
        with open(f"{save_path}/{self.PARAMS_FILE}", "w") as f:
            json.dump(data[1], f, ensure_ascii=False, indent=4)
    
    def _load(self) -> str:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        params = json.load(open(f"{load_path}/{self.PARAMS_FILE}"))
        return load_path, params

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath, version=self._version, protocol=self._protocol
        )
