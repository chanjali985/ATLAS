import pytest
from atlas.tools import Environment


@pytest.fixture(scope="session")
def env():
    return Environment()
