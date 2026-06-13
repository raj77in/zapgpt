"""Tests for multi-prompt, multi-file, and image input support."""

import base64

import pytest

from zapgpt.main import BaseLLMClient, query_llm, resolve_prompt_selection


def test_resolve_multiple_prompts_with_common_base():
    prompts = {
        "common_base": {"system_prompt": "Base"},
        "coding": {"system_prompt": "Code", "model": "coding-model"},
        "review": {
            "system_prompt": "Review",
            "assistant_input": "Be concise",
            "model": "review-model",
        },
    }

    result = resolve_prompt_selection(
        ["coding", "review"],
        prompts,
        base_system_prompt="Custom",
        default_model="default-model",
    )

    assert result == {
        "system_prompt": "Custom\n\nBase\n\nCode\n\nReview",
        "assistant_input": "Be concise",
        "model": "review-model",
    }


def test_resolve_prompts_without_common_base():
    prompts = {
        "common_base": {"system_prompt": "Base"},
        "coding": {"system_prompt": "Code"},
    }

    result = resolve_prompt_selection(
        "coding",
        prompts,
        include_default_prompt=False,
    )

    assert result["system_prompt"] == "Code"


def test_create_prompt_includes_multiple_files(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    client = BaseLLMClient("test-model", files=[str(first), str(second)])
    messages = client.create_prompt("Compare these files")

    assert messages[0] == {"role": "user", "content": "Compare these files"}
    assert messages[1]["content"] == "Filename: first.txt\nFile content:\nalpha"
    assert messages[2]["content"] == "Filename: second.txt\nFile content:\nbeta"


def test_create_prompt_includes_multiple_images(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"png-data")
    second.write_bytes(b"jpg-data")

    client = BaseLLMClient("test-model", images=[str(first), str(second)])
    messages = client.create_prompt("Describe these images")
    content = messages[0]["content"]

    assert content[0] == {"type": "text", "text": "Describe these images"}
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"png-data").decode("ascii")
    )
    assert content[2]["image_url"]["url"] == (
        "data:image/jpeg;base64," + base64.b64encode(b"jpg-data").decode("ascii")
    )


def test_image_rejects_unsupported_file_type(tmp_path):
    image = tmp_path / "image.txt"
    image.write_text("not an image", encoding="utf-8")
    client = BaseLLMClient("test-model", image=str(image))

    with pytest.raises(ValueError, match="Unsupported image file type"):
        client.create_prompt("Describe this")


def test_query_llm_forwards_images_and_prompt_options(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def send_request(self, prompt):
            captured["prompt"] = prompt
            return "response"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(query_llm.__globals__["provider_map"], "openai", FakeClient)

    result = query_llm(
        "Explain",
        provider="openai",
        model="chosen-model",
        image=str(image),
        no_default=True,
    )

    assert result == "response"
    assert captured["model"] == "chosen-model"
    assert captured["images"] == [str(image)]
    assert captured["prompt"] == "Explain"


def test_query_llm_rejects_missing_image(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        query_llm("Explain", image="/missing/diagram.png")
