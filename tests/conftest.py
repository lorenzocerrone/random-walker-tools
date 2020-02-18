import os
import pytest

TEST_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'resources',
)

# common fixtures aimed to reduce the boilerplate in tests
@pytest.fixture
def path_2d_2seeds_16(tmpdir):
    path = os.path.join(tmpdir, '2d_2seeds_16.npy')
    return path