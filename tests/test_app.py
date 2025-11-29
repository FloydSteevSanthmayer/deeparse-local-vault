import pytest
from importlib import import_module

def test_app_imports():
    # Basic smoke test: importing app should not raise an exception
    import_module('app')
    assert True
