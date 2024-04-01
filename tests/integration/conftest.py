import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_path():
    pass
    # This is run before all tests, setup mitsuba3
    # import sys
    # sys.path.insert(0, '../mitsuba3-transient-nlos/ext/mitsuba3/build/python')
