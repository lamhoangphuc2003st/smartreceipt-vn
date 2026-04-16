"""
tests/conftest.py
-----------------
Session-wide pytest configuration.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def disable_rate_limiter():
    """Disable slowapi rate limiting for all tests.

    Tests make many requests from the same IP ('testclient'), which would
    trigger the 10/minute limit and cause unrelated failures.
    """
    from api.limiter import limiter
    limiter.enabled = False
    yield
    limiter.enabled = True
