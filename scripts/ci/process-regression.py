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
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from github import Github

REGRESSION = "regression"
TEST = "test"
MAX_DIFF_CHARS = 40_000

STATUS_FAILURE = "failure"
STATUS_CHANGED = "changed"
STATUS_CANCELLED = "cancelled"
STATUS_SUCCESS = "success"


def run_git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def find_result_dirs(artifacts_dir: Path, prefix: str) -> list[Path]:
    return sorted(p for p in artifacts_dir.glob(f"{prefix}-*") if p.is_dir())


def resolve_merge_base(repo_root: Path, base_ref: str) -> str:
    try:
        run_git(repo_root, "rev-parse", "--verify", base_ref)
    except RuntimeError:
        branch = base_ref.rpartition("/")[2]
        run_git(repo_root, "fetch", "origin", f"{branch}:refs/remotes/origin/{branch}")
    return run_git(repo_root, "merge-base", base_ref, "HEAD").strip()


def get_commit_identity(github_repo, pr_number: int) -> tuple[str, str]:
    """Return (name, email) for the PR author, using their GitHub profile."""
    author = github_repo.get_pull(pr_number).user
    name = author.name or author.login
    email = author.email or f"{author.id}+{author.login}@users.noreply.github.com"
    return name, email


def build_patch(
    *,
    repo_root: Path,
    failure_dirs: list[Path],
    patch_dir: Path,
    author_name: str,
    author_email: str,
) -> list[str]:
    """Copy new baselines into test/, commit, and format-patch. Returns updated file list."""
    updated: list[str] = []
    for result_dir in failure_dirs:
        for subdir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
            dest_dir = repo_root / TEST / subdir.name / REGRESSION
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src_file in sorted(subdir.glob("*")):
                dest_file = dest_dir / src_file.name
                dest_file.write_bytes(src_file.read_bytes())
                updated.append(str(dest_file.relative_to(repo_root)))

    if not updated:
        return updated

    run_git(repo_root, "add", "--", *updated)
    author = f"{author_name} <{author_email}>"
    run_git(
        repo_root,
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
    run_git(repo_root, "format-patch", "-1", "HEAD", "-o", str(patch_dir))
    return updated


def build_diff(repo_root: Path, merge_base: str, pathspec: str) -> tuple[str, str]:
    stat = run_git(
        repo_root, "diff", "--stat", f"{merge_base}..HEAD", "--", pathspec
    ).strip()
    diff = run_git(repo_root, "diff", f"{merge_base}..HEAD", "--", pathspec)
    return stat, diff


def write_comment_body(
    path: Path,
    *,
    status: str,
    updated_files: list[str],
    diff_stat: str,
    diff_text: str,
    actions_run_url: str,
) -> None:
    lines: list[str]
    if status == STATUS_FAILURE:
        lines = [
            "The following regression baselines differ from the recorded expected output:",
            "",
            *(f"- `{f}`" for f in updated_files),
            "",
            "If these changes are expected, apply the attached patch to update them. "
            "If not, this indicates a regression in the physics output that should be "
            "investigated before merging.",
            "",
            f"[View the failing test output in the GitHub Actions run]({actions_run_url})",
        ]
    elif status == STATUS_CHANGED:
        truncated = diff_text
        note = ""
        if len(truncated) > MAX_DIFF_CHARS:
            truncated = truncated[:MAX_DIFF_CHARS]
            note = (
                f"\n_(diff truncated to {MAX_DIFF_CHARS} characters; "
                f"see the [Actions run]({actions_run_url}) for the full diff)_\n"
            )
        lines = [
            "Changes to regression baselines under `test/**/regression/*` were detected "
            "compared to `develop`. These files are marked `linguist-generated`, so GitHub "
            "hides them from the default diff view \u2014 please review below to confirm "
            "the change is intended.",
            "",
            "```",
            diff_stat,
            "```",
            note,
            "<details><summary>Diff</summary>",
            "",
            "```diff",
            truncated,
            "```",
            "</details>",
            "",
            f"[View the GitHub Actions run]({actions_run_url})",
        ]
    else:
        lines = ["Regression tests passed and regression data matches `develop`."]

    path.write_text("\n".join(lines) + "\n")


def write_output(github_output: Optional[str], **kwargs: str) -> None:
    if not github_output:
        return
    with open(github_output, "a") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--base-ref", default="origin/develop")
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
        status = STATUS_FAILURE
        gh = Github(os.environ["GITHUB_TOKEN"])
        github_repo = gh.get_repo(args.repo)
        author_name, author_email = get_commit_identity(github_repo, args.pr_number)
        updated_files = build_patch(
            repo_root=repo_root,
            failure_dirs=failure_dirs,
            patch_dir=args.patch_dir,
            author_name=author_name,
            author_email=author_email,
        )
    else:
        merge_base = resolve_merge_base(repo_root, args.base_ref)
        pathspec = f"{TEST}/**/{REGRESSION}/*"
        diff_stat, diff_text = build_diff(repo_root, merge_base, pathspec)
        if diff_text.strip():
            status = STATUS_CHANGED
        elif not test_result_dirs:
            status = STATUS_CANCELLED
        else:
            status = STATUS_SUCCESS

    write_comment_body(
        args.comment_body_file,
        status=status,
        updated_files=updated_files,
        diff_stat=diff_stat,
        diff_text=diff_text,
        actions_run_url=args.actions_run_url,
    )

    has_patch = bool(updated_files)
    write_output(
        github_output,
        status=status,
        **{"has-patch": "true" if has_patch else "false"},
        **{"patch-dir": str(args.patch_dir) if has_patch else ""},
    )

    if status == STATUS_CANCELLED:
        print("::warning::Regression tests were cancelled or did not report results")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
