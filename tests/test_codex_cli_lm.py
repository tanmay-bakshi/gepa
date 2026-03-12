# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the Codex CLI language-model backend."""

from pathlib import Path
from unittest.mock import patch

import gepa.optimize_anything as oa
from gepa.codex_cli_lm import CodexCLILMConfig, _build_codex_command, parse_codex_cli_spec


class TestParseCodexCliSpec:
    """Tests for ``codex_cli`` backend spec parsing."""

    def test_returns_none_for_non_codex_backend(self) -> None:
        """Non-Codex specs should be ignored."""
        assert parse_codex_cli_spec("openai/gpt-5.1") is None

    def test_uses_defaults_for_short_spec(self) -> None:
        """Bare ``codex_cli`` should use default model and effort."""
        assert parse_codex_cli_spec("codex_cli") == CodexCLILMConfig()

    def test_parses_model_and_effort(self) -> None:
        """Extended ``codex_cli`` specs should override model and effort."""
        parsed = parse_codex_cli_spec("codex_cli:gpt-5.3-codex?reasoning_effort=high")
        assert parsed == CodexCLILMConfig(model="gpt-5.3-codex", reasoning_effort="high")


class TestBuildCodexCommand:
    """Tests for Codex CLI command construction."""

    def test_places_stdin_marker_last(self, tmp_path: Path, monkeypatch) -> None:
        """The stdin marker must remain the final positional argument."""
        codex_home = tmp_path / "codex_home"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(
            '[mcp_servers.docs]\nurl = "https://example.com/mcp"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(codex_home))

        command = _build_codex_command(
            config=CodexCLILMConfig(model="gpt-5.3-codex-spark", reasoning_effort="xhigh"),
            work_dir=Path("/tmp/work"),
            output_path=Path("/tmp/out.txt"),
            image_paths=["/tmp/image-1.png", "/tmp/image-2.png"],
        )

        assert command[-1] == "-"
        assert command.count("-i") == 2
        assert "--disable" in command
        assert "shell_tool" in command
        assert "mcp_servers.docs.enabled=false" in command


class TestMakeBackendLm:
    """Tests for backend LM selection and Codex subprocess execution."""

    def test_codex_cli_spec_uses_codex_backend(self) -> None:
        """``make_backend_lm`` should route ``codex_cli`` specs to Codex."""
        with patch("gepa.optimize_anything.make_codex_cli_lm") as mock_make_codex:
            sentinel = object()
            mock_make_codex.return_value = sentinel
            result = oa.make_backend_lm("codex_cli:gpt-5.3-codex-spark?reasoning_effort=high")
        assert result is sentinel
        parsed_config = mock_make_codex.call_args.args[0]
        assert parsed_config == CodexCLILMConfig(model="gpt-5.3-codex-spark", reasoning_effort="high")

    def test_codex_backend_runs_subprocess_and_reads_output(self, tmp_path: Path, monkeypatch) -> None:
        """The Codex backend should reuse the active Codex home and read the last message file."""
        codex_home = tmp_path / "codex_home"
        codex_home.mkdir()
        (codex_home / "auth.json").write_text('{"token":"test"}', encoding="utf-8")
        (codex_home / "config.toml").write_text(
            '[mcp_servers.openaiDeveloperDocs]\nurl = "https://developers.openai.com/mcp"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(codex_home))

        captured: dict[str, object] = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("mocked output", encoding="utf-8")

            class Result:
                returncode = 0
                stdout = "stdout"
                stderr = "stderr"

            return Result()

        with patch("gepa.codex_cli_lm.subprocess.run", side_effect=fake_run):
            lm = oa.make_backend_lm("codex_cli")
            result = lm("hello world")

        assert result == "mocked output"
        kwargs = captured["kwargs"]
        assert isinstance(kwargs["input"], str)
        assert "hello world" in kwargs["input"]
        assert isinstance(kwargs["env"], dict)
        assert kwargs["env"]["CODEX_HOME"] == str(codex_home)
        command = captured["command"]
        assert "mcp_servers.openaiDeveloperDocs.enabled=false" in command
