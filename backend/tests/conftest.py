import pytest


@pytest.fixture(autouse=True)
def _isolated_intermagnet_cache(tmp_path, monkeypatch):
    """Observatory days are cached on disk between runs. Tests mock the
    network, so each one gets its own empty cache - otherwise mocked data
    would be served to the next test, or to the real app."""
    from app.processing import intermagnet

    monkeypatch.setenv(intermagnet.CACHE_DIR_ENV, str(tmp_path / "intermagnet_cache"))
    monkeypatch.setattr(intermagnet, "_sleep", lambda _s: None)
