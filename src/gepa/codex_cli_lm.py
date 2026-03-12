# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Codex CLI-backed language model helpers for GEPA reflection."""

import base64
import binascii
import mimetypes
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs

from gepa.proposer.reflective_mutation.base import LanguageModel

ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]

_DEFAULT_CODEX_MODEL = "gpt-5.3-codex-spark"
_DEFAULT_REASONING_EFFORT: ReasoningEffort = "xhigh"
_CODEX_PROMPT_PREAMBLE = """You are acting as a pure language model backend inside GEPA.
Use only the instructions and context included below.
Do not inspect files, run commands, browse the web, or ask for more context.
Return only the response requested by the prompt."""


@dataclass(frozen=True)
class CodexCLILMConfig:
    """Configuration for the Codex CLI reflection backend.

    :ivar model: Codex model name to invoke.
    :ivar reasoning_effort: Reasoning effort passed to Codex.
    :ivar executable: Codex CLI executable name or path.
    :ivar timeout_seconds: Maximum subprocess runtime in seconds.
    """

    model: str = _DEFAULT_CODEX_MODEL
    reasoning_effort: ReasoningEffort = _DEFAULT_REASONING_EFFORT
    executable: str = "codex"
    timeout_seconds: float = 600.0


def parse_codex_cli_spec(spec: str) -> CodexCLILMConfig | None:
    """Parse a ``codex_cli`` backend spec.

    Supported forms are:

    - ``codex_cli``
    - ``codex_cli:<model>``
    - ``codex_cli:<model>?reasoning_effort=<effort>``
    - ``codex_cli?reasoning_effort=<effort>``

    :param spec: Backend specification string.
    :returns: Parsed configuration, or ``None`` when the string is not a
        Codex CLI backend specification.
    :raises ValueError: If the query string is invalid.
    """

    if spec == "codex_cli":
        return CodexCLILMConfig()

    if spec.startswith("codex_cli:") is False and spec.startswith("codex_cli?") is False:
        return None

    model: str = _DEFAULT_CODEX_MODEL
    query_string: str = ""

    if spec.startswith("codex_cli:"):
        remainder = spec[len("codex_cli:") :]
        if "?" in remainder:
            model_part, query_string = remainder.split("?", maxsplit=1)
        else:
            model_part = remainder
        if len(model_part.strip()) > 0:
            model = model_part.strip()
    else:
        query_string = spec[len("codex_cli?") :]

    reasoning_effort: ReasoningEffort = _DEFAULT_REASONING_EFFORT
    if len(query_string) > 0:
        parsed_query = parse_qs(query_string, keep_blank_values=False)
        unknown_keys = set(parsed_query.keys()) - {"reasoning_effort"}
        if len(unknown_keys) > 0:
            raise ValueError(f"Unsupported codex_cli query parameters: {sorted(unknown_keys)!r}")
        values = parsed_query.get("reasoning_effort", [])
        if len(values) > 1:
            raise ValueError("codex_cli spec may include at most one reasoning_effort value.")
        if len(values) == 1:
            value = values[0]
            valid_efforts: tuple[ReasoningEffort, ...] = ("minimal", "low", "medium", "high", "xhigh")
            if value not in valid_efforts:
                raise ValueError(f"Unsupported codex_cli reasoning_effort: {value!r}")
            reasoning_effort = value

    return CodexCLILMConfig(model=model, reasoning_effort=reasoning_effort)


def make_codex_cli_lm(config: CodexCLILMConfig | None = None) -> LanguageModel:
    """Create a GEPA-compatible language model backed by ``codex exec``.

    :param config: Optional Codex CLI configuration. When omitted, GEPA uses
        ``gpt-5.3-codex-spark`` with ``xhigh`` reasoning effort.
    :returns: Language-model callable compatible with GEPA reflection hooks.
    """

    if config is None:
        resolved_config = CodexCLILMConfig()
    else:
        resolved_config = config

    def _lm(prompt: str | list[dict[str, Any]]) -> str:
        return _run_codex_cli_prompt(prompt=prompt, config=resolved_config)

    return _lm


def _run_codex_cli_prompt(prompt: str | list[dict[str, Any]], config: CodexCLILMConfig) -> str:
    """Run ``codex exec`` for a reflection prompt.

    :param prompt: Prompt text or chat-style messages payload.
    :param config: Codex CLI configuration.
    :returns: Final assistant message content.
    :raises FileNotFoundError: If the source Codex auth file is missing.
    :raises RuntimeError: If Codex exits unsuccessfully or produces no final
        assistant message.
    """

    source_codex_home = _resolve_source_codex_home()
    source_auth_path = source_codex_home / "auth.json"
    if source_auth_path.is_file() is False:
        raise FileNotFoundError(
            "Codex CLI auth.json was not found. Run `codex login` before using the codex_cli backend."
        )

    with tempfile.TemporaryDirectory(prefix="gepa-codex-work-") as work_dir_name:
        work_dir = Path(work_dir_name)
        prompt_text, image_paths = _prepare_prompt_payload(prompt=prompt, scratch_dir=work_dir)
        output_path = work_dir / "codex_output.txt"
        command = _build_codex_command(
            config=config,
            work_dir=work_dir,
            output_path=output_path,
            image_paths=image_paths,
        )

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=os.environ.copy(),
            input=prompt_text,
            text=True,
            timeout=config.timeout_seconds,
        )
        if completed.returncode != 0:
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            raise RuntimeError(
                "Codex CLI reflection backend failed with exit code "
                f"{completed.returncode}. stdout={stdout!r} stderr={stderr!r}"
            )

        if output_path.is_file() is False:
            raise RuntimeError("Codex CLI reflection backend did not produce an output-last-message file.")

        output_text = output_path.read_text(encoding="utf-8").strip()
        if len(output_text) == 0:
            raise RuntimeError("Codex CLI reflection backend produced an empty response.")
        return output_text


def _resolve_source_codex_home() -> Path:
    """Resolve the source ``CODEX_HOME`` from which auth should be copied.

    :returns: Path to the source Codex home directory.
    """

    env_value = os.environ.get("CODEX_HOME")
    if env_value is not None and len(env_value.strip()) > 0:
        return Path(env_value).expanduser()
    return Path.home() / ".codex"


def _configured_mcp_server_names(codex_home: Path) -> list[str]:
    """Read configured MCP server names from the user's Codex config.

    :param codex_home: Active Codex home directory.
    :returns: Configured MCP server names.
    """

    config_path = codex_home / "config.toml"
    if config_path.is_file() is False:
        return []

    server_names: list[str] = []
    config_text = config_path.read_text(encoding="utf-8")
    simple_pattern = re.compile(r"^\[mcp_servers\.([A-Za-z0-9_-]+)\]$", re.MULTILINE)
    quoted_pattern = re.compile(r'^\[mcp_servers\."([^"]+)"\]$', re.MULTILINE)
    server_names.extend(simple_pattern.findall(config_text))
    server_names.extend(quoted_pattern.findall(config_text))
    return server_names


def _prepare_prompt_payload(
    prompt: str | list[dict[str, Any]],
    scratch_dir: Path,
) -> tuple[str, list[str]]:
    """Render the prompt into Codex CLI stdin text and optional images.

    :param prompt: GEPA prompt payload.
    :param scratch_dir: Scratch directory for decoded image files.
    :returns: A tuple of rendered prompt text and image paths.
    :raises TypeError: If the prompt payload is malformed.
    :raises ValueError: If an image payload is unsupported.
    """

    if isinstance(prompt, str):
        return f"{_CODEX_PROMPT_PREAMBLE}\n\n{prompt}", []

    rendered_sections: list[str] = []
    image_paths: list[str] = []
    for index, message in enumerate(prompt, start=1):
        role = message.get("role")
        if isinstance(role, str) is False:
            raise TypeError("Codex CLI prompts must contain string message roles.")
        role_text = str(role)

        content = message.get("content")
        message_text, message_images = _render_message_content(
            content=content,
            scratch_dir=scratch_dir,
            image_offset=len(image_paths),
        )
        image_paths.extend(message_images)
        rendered_sections.append(f"## {role_text.upper()} MESSAGE {index}\n{message_text}")

    rendered_prompt = "\n\n".join(rendered_sections)
    return f"{_CODEX_PROMPT_PREAMBLE}\n\n{rendered_prompt}", image_paths


def _render_message_content(
    content: Any,
    scratch_dir: Path,
    image_offset: int,
) -> tuple[str, list[str]]:
    """Render one chat message payload.

    :param content: Message content from the GEPA prompt payload.
    :param scratch_dir: Scratch directory for decoded image files.
    :param image_offset: Number of images already written for earlier messages.
    :returns: A tuple of rendered message text and decoded image paths.
    :raises TypeError: If the content payload is malformed.
    :raises ValueError: If an image payload is unsupported.
    """

    if isinstance(content, str):
        return content, []

    if isinstance(content, list) is False:
        raise TypeError("Codex CLI prompts must contain string content or OpenAI-style content-part arrays.")

    rendered_parts: list[str] = []
    image_paths: list[str] = []
    for part in content:
        if isinstance(part, dict) is False:
            raise TypeError("Codex CLI content-part arrays must contain dictionaries.")

        part_type = part.get("type")
        if part_type == "text":
            text_value = part.get("text")
            if isinstance(text_value, str) is False:
                raise TypeError("Codex CLI text content-parts must provide a string `text` field.")
            rendered_parts.append(text_value)
            continue

        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict) is False:
                raise TypeError("Codex CLI image_url content-parts must provide a dict `image_url` field.")
            url_value = image_url.get("url")
            if isinstance(url_value, str) is False:
                raise TypeError("Codex CLI image_url content-parts must provide a string `url` field.")
            image_number = image_offset + len(image_paths) + 1
            image_path = _materialize_image_url(
                image_url=url_value,
                scratch_dir=scratch_dir,
                image_number=image_number,
            )
            image_paths.append(image_path)
            rendered_parts.append(f"[Attached image {image_number}]")
            continue

        raise ValueError(f"Unsupported Codex CLI content-part type: {part_type!r}")

    return "\n".join(rendered_parts), image_paths


def _materialize_image_url(image_url: str, scratch_dir: Path, image_number: int) -> str:
    """Convert an OpenAI ``image_url`` payload into a local file for Codex.

    :param image_url: Image URL or ``data:`` URI.
    :param scratch_dir: Scratch directory for decoded image files.
    :param image_number: Stable image index for the temporary filename.
    :returns: Local image file path.
    :raises ValueError: If the image URL format is unsupported.
    """

    if image_url.startswith("data:") is False:
        candidate_path = Path(image_url).expanduser()
        if candidate_path.is_file() is False:
            raise ValueError(
                "Codex CLI only supports data: image URLs or existing local file paths in message content."
            )
        return str(candidate_path)

    header, encoded_data = image_url.split(",", maxsplit=1)
    media_type = header[5:].split(";", maxsplit=1)[0]
    file_suffix = mimetypes.guess_extension(media_type)
    if file_suffix is None:
        file_suffix = ".bin"
    image_path = scratch_dir / f"codex_prompt_image_{image_number}{file_suffix}"
    try:
        image_bytes = base64.b64decode(encoded_data, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 image payload for codex_cli backend.") from exc
    image_path.write_bytes(image_bytes)
    return str(image_path)


def _build_codex_command(
    config: CodexCLILMConfig,
    work_dir: Path,
    output_path: Path,
    image_paths: list[str],
) -> list[str]:
    """Build the ``codex exec`` command line.

    :param config: Codex CLI configuration.
    :param work_dir: Empty working directory for the Codex subprocess.
    :param output_path: Path for ``--output-last-message``.
    :param image_paths: Local image files to attach.
    :returns: Command argument vector.
    """

    command: list[str] = [
        config.executable,
        "-a",
        "never",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-C",
        str(work_dir),
        "--output-last-message",
        str(output_path),
        "-m",
        config.model,
        "--disable",
        "shell_tool",
        "-c",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "-c",
        "notify=[]",
        "-c",
        "suppress_unstable_features_warning=true",
    ]
    for server_name in _configured_mcp_server_names(_resolve_source_codex_home()):
        command.extend(["-c", f"mcp_servers.{server_name}.enabled=false"])
    for image_path in image_paths:
        command.extend(["-i", image_path])
    command.append("-")
    return command
