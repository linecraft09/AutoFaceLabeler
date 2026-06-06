from abc import ABCMeta
from pathlib import Path

import pytest
import yaml

from aflutils.config_loader import ConfigLoader
from annotators.base_annotator import BaseAnnotator


CONFIG_PATH = Path("config/annotation_config.yaml")


def load_annotation_config():
    return ConfigLoader(CONFIG_PATH, load_env=False).as_dict()


def test_config_file_exists():
    assert CONFIG_PATH.exists()
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)


def test_required_sections():
    data = load_annotation_config()

    assert {"deepface", "qwen_vl", "dwpose", "whisper"} <= set(data)


def test_default_values():
    loader = ConfigLoader(CONFIG_PATH, load_env=False)

    assert loader.get("annotation.target_annotated") == 200
    assert loader.get("annotation.device") == "cuda"


def test_deepface_config_structure():
    deepface = load_annotation_config()["deepface"]

    assert deepface["enabled"] is True
    assert deepface["device"] == "cuda"
    assert isinstance(deepface["actions"], list)
    assert {"age", "race"} <= set(deepface["actions"])
    assert deepface["sample_frames"] > 0


def test_qwen_vl_config_structure():
    qwen_vl = load_annotation_config()["qwen_vl"]

    assert qwen_vl["enabled"] is True
    assert qwen_vl["model"]
    assert qwen_vl["api_key_env"]
    assert qwen_vl["max_frames"] > 0
    assert 0 <= qwen_vl["temperature"] <= 1


def test_dwpose_config_structure():
    dwpose = load_annotation_config()["dwpose"]

    assert dwpose["enabled"] is True
    assert dwpose["device"] == "cuda"
    assert dwpose["model_path"]
    assert dwpose["batch_size"] > 0
    assert dwpose["clarity"]["crop_padding"] >= 0


def test_whisper_config_structure():
    whisper = load_annotation_config()["whisper"]

    assert whisper["enabled"] is True
    assert whisper["device"] == "cuda"
    assert whisper["model_name"]
    assert whisper["compute_type"]
    assert isinstance(whisper["segments"], dict)


def test_annotation_pipeline_defaults():
    annotation = load_annotation_config()["annotation"]

    assert annotation["target_annotated"] == 200
    assert annotation["device"] == "cuda"
    assert annotation["batch_size"] > 0
    assert annotation["output_dir"]


def test_annotators_package_imports():
    import annotators

    assert hasattr(annotators, "BaseAnnotator")
    assert annotators.BaseAnnotator is BaseAnnotator


def test_base_annotator_abstract():
    assert isinstance(BaseAnnotator, ABCMeta)
    with pytest.raises(TypeError):
        BaseAnnotator({"device": "cuda"})


def test_base_annotator_label_name():
    assert "label_name" in BaseAnnotator.__abstractmethods__
