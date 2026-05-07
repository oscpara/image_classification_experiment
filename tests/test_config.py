from src import config


def test_get_database_url_prefers_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///env.db")

    assert config.get_database_url() == "sqlite:///env.db"


def test_get_database_url_reads_config_ini(tmp_path, monkeypatch):
    config_path = tmp_path / "config.ini"
    config_path.write_text("[database]\nurl = sqlite:///config.db\n", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)

    assert config.get_database_url() == "sqlite:///config.db"


def test_get_database_url_reads_percent_encoded_config_value(tmp_path, monkeypatch):
    database_url = "postgresql+psycopg://user:p%40ss@localhost:5432/myapp"
    config_path = tmp_path / "config.ini"
    config_path.write_text(f"[database]\nurl = {database_url}\n", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)

    assert config.get_database_url() == database_url
