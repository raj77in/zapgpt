"""Tests for the public zapgpt package interface."""


def test_public_api_exports():
    import zapgpt

    assert callable(zapgpt.main)
    assert callable(zapgpt.query_llm)
    assert set(zapgpt.__all__) == {"main", "query_llm"}


def test_version_and_metadata():
    import zapgpt

    assert isinstance(zapgpt.__version__, str)
    assert isinstance(zapgpt.__author__, str)
    assert isinstance(zapgpt.__email__, str)
