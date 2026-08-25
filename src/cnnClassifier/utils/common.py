import os
import json
import base64
import joblib
import yaml

from box.exceptions import BoxValueError
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path
from typing import Any

from cnnClassifier import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns ConfigBox.

    Args:
        path_to_yaml (Path): Path to YAML file.

    Raises:
        ValueError: If YAML file is empty.

    Returns:
        ConfigBox: YAML content as ConfigBox.
    """

    try:
        with open(
            path_to_yaml,
            mode="r",
            encoding="utf-8"
        ) as yaml_file:

            content = yaml.safe_load(yaml_file)

            logger.info(
                f"yaml file: {path_to_yaml} loaded successfully"
            )

            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("yaml file is empty")

    except Exception as e:
        raise e


@ensure_annotations
def create_directories(
    path_to_directories: list,
    verbose=True
):
    """
    Create directories.

    Args:
        path_to_directories (list): List of directory paths.
        verbose (bool): Log created directories.
    """

    for path in path_to_directories:
        os.makedirs(
            path,
            exist_ok=True
        )

        if verbose:
            logger.info(
                f"created directory at: {path}"
            )


@ensure_annotations
def save_json(
    path: Path,
    data: dict
):
    """
    Save dictionary as JSON.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    logger.info(
        f"json file saved at: {path}"
    )


@ensure_annotations
def load_json(
    path: Path
) -> ConfigBox:
    """
    Load JSON file.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        content = json.load(f)

    logger.info(
        f"json file loaded successfully from: {path}"
    )

    return ConfigBox(content)


@ensure_annotations
def save_bin(
    data: Any,
    path: Path
):
    """
    Save binary data.
    """

    joblib.dump(
        value=data,
        filename=path
    )

    logger.info(
        f"binary file saved at: {path}"
    )


@ensure_annotations
def load_bin(
    path: Path
) -> Any:
    """
    Load binary data.
    """

    data = joblib.load(path)

    logger.info(
        f"binary file loaded from: {path}"
    )

    return data


@ensure_annotations
def get_size(
    path: Path
) -> str:
    """
    Get file size in KB.
    """

    size_in_kb = round(
        os.path.getsize(path) / 1024
    )

    return f"~ {size_in_kb} KB"


def decodeImage(
    imgstring,
    fileName
):
    imgdata = base64.b64decode(
        imgstring
    )

    with open(
        fileName,
        "wb"
    ) as f:

        f.write(imgdata)


def encodeImageIntoBase64(
    croppedImagePath
):
    with open(
        croppedImagePath,
        "rb"
    ) as f:

        return base64.b64encode(
            f.read()
        )