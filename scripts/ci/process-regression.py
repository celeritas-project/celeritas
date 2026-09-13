#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Analyze downloaded regression artifacts and prepare a PR comment.

This is the first of two scripts used by the ``pull_request_completed``
workflow's ``report-regression`` job:

- ``process-regression.py`` (this script) inspects the artifacts downloaded
  from the ``pull_request`` workflow run, classifies the outcome, and (for a
  ``failure``) stages an updated set of regression baselines as a local git
  commit and ``git format-patch`` file. It writes the PR comment body to a
  file and emits ``status``/``has-patch``/``patch-dir`` via ``GITHUB_OUTPUT``.
- ``post-regression-comment.py`` is run afterwards (once any patch has been
  uploaded as a workflow artifact) to actually create/update the PR comment.

Splitting the two is necessary because only a real ``actions/upload-artifact``
step can produce the patch download URL referenced by the comment.

.. important::
   This script itself must only ever be checked out from the trusted base
   branch (never from the PR being tested). The PR's commit (``--pr-ref``) is
   only ever read as inert data (``git diff``/``git worktree``) — it is never
   checked out as the working tree HEAD and nothing from it is executed.
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from github import Github

REGRESSION = "regression"
TEST = "test"
MAX_DIFF_CHARS = 40_000


class Status(StrEnum):
    FAILURE = "failure"
    CHANGED = "changed"
    SUCCESS = "success"
    CANCELLED = "cancelled"


class LogLevel(StrEnum):
    DEBUG = "debug"
    NOTICE = "notice"
    WARNING = "warning"


def log(level: str | LogLevel, what: str) -> None:
    """Emit a GitHub Actions log annotation with caller filename and line number."""
    if not isinstance(level, LogLevel):
        level = LogLevel(level.lower())

    if level not in LogLevel:
        raise ValueError(f"unsupported log level: {level!r}")

    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    filename = "<unknown>"
    lineno = 0
    if caller is not None:
        filename = caller.f_code.co_filename
        lineno = caller.f_lineno

    msg = str(what).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} file={filename},line={lineno}::{msg}")


def run_git(repo_root: Path, *args: str, check: bool = True) -> str:
    log(LogLevel.DEBUG, f"Calling git {args!r}")
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def find_result_dirs(artifacts_dir: Path, prefix: str) -> list[Path]:
    return sorted(p for p in artifacts_dir.glob(f"{prefix}-*") if p.is_dir())


def get_commit_identity(github_repo, pr_number: int) -> tuple[str, str]:
    """Return (name, email) for the PR author, using their GitHub profile."""
    author = github_repo.get_pull(pr_number).user
    name = author.name or author.login
    email = author.email or f"{author.id}+{author.login}@users.noreply.github.com"
    return name, email


def build_patch(
    *,
    repo_root: Path,
    pr_ref: str,
    failure_dirs: list[Path],
    patch_dir: Path,
    author_name: str,
    author_email: str,
) -> list[str]:
    """Materialize the PR commit in a scratch worktree, update baselines, and format-patch.

    The worktree is only ever read from and written to by this function (never executed),
    so materializing the untrusted PR commit here is safe.
    """
    updated: list[str] = []
    for result_dir in failure_dirs:
        for subdir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
            for src_file in sorted(subdir.glob("*")):
                updated.append(
                    str(Path(TEST) / subdir.name / REGRESSION / src_file.name)
                )

    if not updated:
        log(LogLevel.DEBUG, "No regression output changes were found")
        return updated

    with tempfile.TemporaryDirectory() as worktree_str:
        worktree = Path(worktree_str)
        run_git(repo_root, "worktree", "add", "--detach", str(worktree), pr_ref)
        try:
            for result_dir in failure_dirs:
                for subdir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
                    dest_dir = worktree / TEST / subdir.name / REGRESSION
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for src_file in sorted(subdir.glob("*")):
                        (dest_dir / src_file.name).write_bytes(src_file.read_bytes())

            run_git(worktree, "add", "--", *updated)
            author = f"{author_name} <{author_email}>"
            run_git(
                worktree,
                "-c",
                f"user.name={author_name}",
                "-c",
                f"user.email={author_email}",
                "commit",
                f"--author={author}",
                "-m",
                "Update regression baselines from CI",
            )
            patch_dir.mkdir(parents=True, exist_ok=True)
            run_git(
                worktree, "format-patch", "-1", "HEAD", "-o", str(patch_dir.resolve())
            )
        finally:
            run_git(
                repo_root, "worktree", "remove", "--force", str(worktree), check=False
            )
    return updated


def make_comment_body(
    *,
    status: str,
    updated_files: list[str],
    diff_stat: str,
    diff_text: str,
    actions_run_url: str,
) -> str:
    if status == Status.FAILURE:
        updated = "\n".join(f"- `{f}`" for f in updated_files)
        return f"""\
The following regression baselines differ from the recorded expected output:
{updated}

If these changes are expected, apply the attached patch to update them.
If not, this indicates a regression in the physics output that should be investigated before merging.

[View the failing test output in the GitHub Actions run]({actions_run_url})
"""
    if status == Status.CHANGED:
        truncated = diff_text
        note = ""
        if len(truncated) > MAX_DIFF_CHARS:
            truncated = truncated[:MAX_DIFF_CHARS]
            note = (
                f"\n_(diff truncated to {MAX_DIFF_CHARS} characters; "
                f"see the [Actions run]({actions_run_url}) for the full diff)_\n"
            )
        return f"""\
Changes to regression baselines under `test/**/regression/*` were detected compared to `develop`.
These files are marked `linguist-generated`, so GitHub hides them from the default diff view:
**please review below** to confirm the change is intended.

```
{diff_stat}
```
{note}<details><summary>Diff</summary>

```diff
{truncated}
```
</details>

[View the GitHub Actions run]({actions_run_url})
"""

    assert status == Status.SUCCESS
    return "Regression tests passed and regression data matches `develop`.\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument(
        "--pr-ref",
        required=True,
        help="local ref/sha for the (untrusted) PR head commit, fetched as inert data only",
    )
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="trusted base ref to diff against (default: the checked-out base branch)",
    )
    parser.add_argument("--actions-run-url", required=True)
    parser.add_argument("--comment-body-file", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    github_output = os.environ.get("GITHUB_OUTPUT")

    failure_dirs = find_result_dirs(args.artifacts_dir, "regression-results")
    test_result_dirs = find_result_dirs(args.artifacts_dir, "test-results-regression")

    updated_files: list[str] = []
    diff_stat = ""
    diff_text = ""

    if failure_dirs:
        log(LogLevel.DEBUG, "Found regression failures")
        status = Status.FAILURE
        gh = Github(os.environ["GITHUB_TOKEN"])
        github_repo = gh.get_repo(args.repo)
        author_name, author_email = get_commit_identity(github_repo, args.pr_number)
        updated_files = build_patch(
            repo_root=repo_root,
            pr_ref=args.pr_ref,
            failure_dirs=failure_dirs,
            patch_dir=args.patch_dir,
            author_name=author_name,
            author_email=author_email,
        )
    else:
        log(LogLevel.DEBUG, "Comparing against merge base")
        pr_ref = args.pr_ref
        merge_base = run_git(repo_root, "merge-base", args.base_ref, pr_ref).strip()
        pathspec = f"{TEST}/**/{REGRESSION}/*"
        diff_stat = run_git(
            repo_root, "diff", "--stat", f"{merge_base}..{pr_ref}", "--", pathspec
        ).strip()
        log(
            LogLevel.DEBUG,
            "Diff: " + ("<no change>" if not diff_stat else diff_stat.splitlines()[-1]),
        )
        diff_stat = run_git(
            repo_root, "diff", f"{merge_base}..{pr_ref}", "--", pathspec
        )
        if diff_text.strip():
            status = Status.CHANGED
        elif not test_result_dirs:
            status = Status.CANCELLED
        else:
            status = Status.SUCCESS

    args.comment_body_file.write_text(
        make_comment_body(
            status=status,
            updated_files=updated_files,
            diff_stat=diff_stat,
            diff_text=diff_text,
            actions_run_url=args.actions_run_url,
        )
    )

    updates = [
        f"{k}={v}\n"
        for (k, v) in [
            ("status", status),
            ("updated-files", json.dumps(updated_files)),
            ("patch-dir", str(args.patch_dir) if updated_files else ""),
        ]
    ]
    for line in updates:
        log(LogLevel.NOTICE, line.rstrip())

    if github_output is not None:
        log(LogLevel.DEBUG, f"Writing git output to {github_output!r}")
        with open(github_output, "a") as f:
            f.writelines(updates)

    if status == Status.CANCELLED:
        log(
            LogLevel.WARNING,
            "Regression tests were cancelled or did not report results",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
