"""Pytest behavior for the intentionally empty initial test suite."""


def pytest_sessionfinish(session, exitstatus):
    """Treat an empty configured suite as a successful scaffold check."""

    if exitstatus == 5:
        session.exitstatus = 0
