#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Check that the Claude Code plugin refers only to things that exist.

The user-facing plugin in ``contrib/claude-plugin`` describes Celeritas
files, environment variables, setup options, UI commands, and classes. This
check fails when the plugin's Markdown refers to any of them after they have
been renamed or removed, so that the plugin is updated together with the
code. It also validates the plugin and marketplace manifests and the skill
front matter.

Run from anywhere: ``python3 scripts/ci/check-claude-plugin.py``.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = ROOT / "contrib" / "claude-plugin"
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
EVALS_DIRNAME = "evals"

ENVIRONMENT_DOC = ROOT / "doc" / "usage" / "execution" / "environment.rst"
SETUP_OPTIONS = ROOT / "src" / "accel" / "SetupOptions.hh"
UI_MESSENGER = ROOT / "src" / "accel" / "SetupOptionsMessenger.cc"

# Repository-relative paths, e.g. `src/accel/SetupOptions.hh` or
# `src/accel/{SharedParams,LocalTransporter}.hh`; not preceded by a path or
# URL character so that `<prefix>/lib/cmake/...` and URLs are skipped
REPO_PATH_RE = re.compile(
    r"(?<![\w/.${}<>-])"
    r"((?:src|app|example|doc|cmake|scripts|test)/[\w.+\-/{},*]*[\w*}/])"
)
# Installed-header style paths, e.g. `accel/SetupOptions.hh`, which live
# under src/ in the repository
INSTALLED_PATH_RE = re.compile(
    r"(?<![\w/.${}<>-])"
    r"((?:corecel|geocel|orange|celeritas|accel|ddceler|larceler)/"
    r"[\w.+\-/{},*]*\.(?:hh|cc|cu|fcl))\b"
)
ENV_VAR_RE = re.compile(r"\bCELER_[A-Z0-9_]+\b")
OPTION_RE = re.compile(r"\bopts\.((?:\w+\.)*\w+)")
UI_CMD_RE = re.compile(r"(?<![\w/])/celer/[\w/]*")
QUALIFIED_RE = re.compile(r"\b(?:celeritas|inp)::(?:\w+::)*(\w+)")
MD_LINK_RE = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
SKILL_DIR_RE = re.compile(r"\$\{CLAUDE_SKILL_DIR\}/([\w./-]+)")
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


class Checker:
    def __init__(self):
        self.errors = []

    def error(self, path, lineno, msg):
        rel = path.relative_to(ROOT) if path.is_absolute() else path
        loc = f"{rel}:{lineno}" if lineno else f"{rel}"
        self.errors.append(f"{loc}: {msg}")


def expand_braces(pattern):
    """Expand shell-style ``{a,b}`` groups (non-nested) into a list."""
    match = re.search(r"\{([^{}]*)\}", pattern)
    if not match:
        return [pattern]
    head, tail = pattern[: match.start()], pattern[match.end() :]
    result = []
    for alt in match.group(1).split(","):
        result.extend(expand_braces(head + alt + tail))
    return result


def repo_path_exists(pattern):
    for candidate in expand_braces(pattern.rstrip("/")):
        if any(c in candidate for c in "*?["):
            if not any(ROOT.glob(candidate)):
                return False
        elif not (ROOT / candidate).exists():
            return False
    return True


def load_env_vars():
    """Names documented in environment.rst plus CELER_ preprocessor macros."""
    names = set(ENV_VAR_RE.findall(ENVIRONMENT_DOC.read_text(encoding="utf-8")))
    define_re = re.compile(r"^\s*#\s*define\s+(CELER_\w+)", re.MULTILINE)
    for header in (ROOT / "src").rglob("*.hh"):
        names.update(define_re.findall(header.read_text(encoding="utf-8")))
    return names


def load_option_members():
    """Data member names of the SetupOptions family of structs."""
    text = SETUP_OPTIONS.read_text(encoding="utf-8")
    # Declarations like `Type name;`, `Type name{init};`, `Type name = x;`
    # with arbitrary template/function types (but not comments or calls)
    member_re = re.compile(
        r"^\s+(?!return\b)[^;/{}\n]*?[\s>*&](\w+)\s*(?:\{[^;\n]*\}|=[^;\n]*)?;",
        re.MULTILINE,
    )
    return set(member_re.findall(text))


def load_ui_commands():
    """Directories and full command paths from SetupOptionsMessenger."""
    text = UI_MESSENGER.read_text(encoding="utf-8")
    token_re = re.compile(
        r'CelerDirectory\(\s*"([^"]+)"|add_cmd\(\s*[^,()]+,\s*"([^"]+)"'
    )
    commands = set()
    directory = None
    for match in token_re.finditer(text):
        if match.group(1):
            directory = match.group(1)
            commands.add(directory)
        elif directory:
            commands.add(directory + match.group(2))
    return commands


def load_identifiers():
    """All identifiers appearing in Celeritas headers."""
    words = set()
    word_re = re.compile(r"\b[A-Za-z_]\w*\b")
    for header in (ROOT / "src").rglob("*.hh"):
        words.update(word_re.findall(header.read_text(encoding="utf-8")))
    return words


def check_markdown(checker, path, refs):
    text = path.read_text(encoding="utf-8")
    parts = path.relative_to(PLUGIN_DIR).parts
    skill_dir = None
    if len(parts) >= 3 and parts[0] == "skills":
        skill_dir = PLUGIN_DIR / parts[0] / parts[1]

    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        # Markdown links are only meaningful outside code
        prose = "" if in_fence else INLINE_CODE_RE.sub("", line)

        for match in REPO_PATH_RE.finditer(line):
            token = match.group(1).rstrip(".,:;")
            if not repo_path_exists(token):
                checker.error(path, lineno, f"repository path '{token}' does not exist")
        for match in INSTALLED_PATH_RE.finditer(line):
            token = match.group(1)
            if not repo_path_exists("src/" + token):
                checker.error(
                    path, lineno, f"header path '{token}' does not exist under src/"
                )
        for name in ENV_VAR_RE.findall(line):
            if name not in refs["env"]:
                checker.error(
                    path,
                    lineno,
                    f"'{name}' is neither documented in "
                    f"{ENVIRONMENT_DOC.relative_to(ROOT)} nor a CELER_ macro",
                )
        for match in OPTION_RE.finditer(line):
            for member in match.group(1).split("."):
                if member not in refs["options"]:
                    checker.error(
                        path,
                        lineno,
                        f"'opts.{match.group(1)}': '{member}' is not a member "
                        f"in {SETUP_OPTIONS.relative_to(ROOT)}",
                    )
        for match in UI_CMD_RE.finditer(line):
            cmd = match.group(0)
            if cmd not in refs["ui"]:
                checker.error(
                    path,
                    lineno,
                    f"UI command '{cmd}' is not defined in "
                    f"{UI_MESSENGER.relative_to(ROOT)}",
                )
        for match in QUALIFIED_RE.finditer(line):
            name = match.group(1)
            if name not in refs["identifiers"]:
                checker.error(
                    path,
                    lineno,
                    f"'{match.group(0)}': '{name}' does not appear in any src/ header",
                )
        for target in MD_LINK_RE.findall(prose):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / target).exists():
                checker.error(path, lineno, f"broken relative link '{target}'")
        for target in SKILL_DIR_RE.findall(line):
            if skill_dir is None or not (skill_dir / target).exists():
                checker.error(
                    path, lineno, f"'${{CLAUDE_SKILL_DIR}}/{target}' does not exist"
                )


def check_front_matter(checker, path, kind):
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(text)
    if not match:
        checker.error(path, 1, f"{kind} must start with YAML front matter")
        return
    fields = dict(
        line.split(":", 1)
        for line in match.group(1).splitlines()
        if re.match(r"^[\w-]+:", line)
    )
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    if not fields.get("description"):
        checker.error(path, 1, f"{kind} front matter needs a description")
    expected = path.parent.name if kind == "skill" else path.stem
    if fields.get("name", expected) != expected:
        checker.error(
            path, 1, f"{kind} name '{fields['name']}' should match '{expected}'"
        )


def load_json(checker, path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        checker.error(path, 0, f"cannot load JSON: {e}")
        return None


def check_manifests(checker):
    marketplace = load_json(checker, MARKETPLACE)
    manifest_path = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
    manifest = load_json(checker, manifest_path)
    if not (marketplace and manifest):
        return
    for key in ("name", "owner", "plugins"):
        if key not in marketplace:
            checker.error(MARKETPLACE, 0, f"missing required key '{key}'")
    sources = {}
    for entry in marketplace.get("plugins", []):
        source = entry.get("source")
        if isinstance(source, str):
            if ".." in source or not (ROOT / source).is_dir():
                checker.error(
                    MARKETPLACE, 0, f"plugin source '{source}' is not a directory"
                )
            sources[entry.get("name")] = (ROOT / source).resolve()
    if sources.get(manifest.get("name")) != PLUGIN_DIR.resolve():
        checker.error(
            MARKETPLACE,
            0,
            f"no marketplace entry named '{manifest.get('name')}' points to "
            f"{PLUGIN_DIR.relative_to(ROOT)}",
        )
    if "version" in manifest:
        checker.error(
            manifest_path,
            0,
            "do not set 'version': the plugin is versioned by git commit, so a "
            "fixed version would stop users from receiving updates",
        )


def check_evals(checker):
    """Check the structure of the eval suite (cases, graders, scaffolds)."""
    evals_dir = PLUGIN_DIR / EVALS_DIRNAME
    if not evals_dir.is_dir():
        return
    scaffold_re = re.compile(r"^\s*scaffold_script:\s*['\"]?([^'\"\s]+)", re.MULTILINE)
    for case in sorted(p for p in evals_dir.iterdir() if p.is_dir()):
        if case.name.startswith("_") or case.name == "results":
            continue
        case_yaml = case / "case.yaml"
        if not ((case / "prompt.md").is_file() or case_yaml.is_file()):
            checker.error(case, 0, "eval case needs prompt.md or case.yaml")
            continue
        if not any((case / "graders").glob("*.md")) and not (
            case_yaml.is_file() and "graders:" in case_yaml.read_text(encoding="utf-8")
        ):
            checker.error(case, 0, "eval case has no graders")
        if case_yaml.is_file():
            for script in scaffold_re.findall(case_yaml.read_text(encoding="utf-8")):
                path = case / script
                if not path.is_file():
                    checker.error(
                        case_yaml, 0, f"scaffold_script '{script}' does not exist"
                    )
                elif not os.access(path, os.X_OK):
                    checker.error(path, 0, "scaffold script is not executable")


def main():
    checker = Checker()
    check_manifests(checker)
    check_evals(checker)

    refs = {
        "env": load_env_vars(),
        "options": load_option_members(),
        "ui": load_ui_commands(),
        "identifiers": load_identifiers(),
    }

    for skill in sorted(PLUGIN_DIR.glob("skills/*/SKILL.md")):
        check_front_matter(checker, skill, "skill")
    for agent in sorted(PLUGIN_DIR.glob("agents/*.md")):
        check_front_matter(checker, agent, "agent")
    for md in sorted(PLUGIN_DIR.rglob("*.md")):
        # Eval prompts and graders refer to files in the eval workspace
        if EVALS_DIRNAME in md.relative_to(PLUGIN_DIR).parts[:1]:
            continue
        check_markdown(checker, md, refs)

    for msg in checker.errors:
        print(msg)
    if checker.errors:
        print(
            f"\n{len(checker.errors)} problem(s) in the Claude Code plugin. "
            "Update contrib/claude-plugin to match the code."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
