from pathlib import Path
import pytest
import shutil
import os


@pytest.fixture
def temp_dir():
    temporary_directory = os.path.join(Path.home(), '.pandora_test')
    if os.path.exists(temporary_directory):
        shutil.rmtree(temporary_directory)
    os.makedirs(temporary_directory)

    return temporary_directory
