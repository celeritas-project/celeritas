#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Analyze downloaded regression artifacts and prepare a PR comment.

The comment is generated to stdout or, optionally, to an output file.

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
import json
import os
import sys
import tempfile
import textwrap
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from _regression_utils import LogLevel, Status, log, run_git

# See build-regression.yml for artifact names and CMakeLists/run-output-regression for subdir names
REGRESSION_OUTPUT_ARTIFACT = "regression-output"
TEST_OUTPUT_ARTIFACT = "test-results-regression"
REGRESSION_SUBDIR = "regression"
TEST_SUBDIR = "test"
MAX_DIFF_CHARS = 10_000


def find_result_dirs(artifacts_dir: Path, prefix: str) -> list[Path]:
    result = sorted(p for p in artifacts_dir.glob(f"{prefix}-*") if p.is_dir())
    count = len(result)
    log(LogLevel.DEBUG, f"Found {count} results in {artifacts_dir} matching {prefix}")
    return result


def process_failures(
    *,
    failure_dirs: list[Path],
    author: str,
    repo_root: Path,
    pr_ref: str,
    patch_dir: Path,
    actions_run_url: str,
) -> tuple[Status, str | None, dict[str, str]]:
    """Materialize the PR commit in a scratch worktree, update baselines, and format-patch.

    The worktree is only ever read from and written to by this function (never executed),
    so materializing the untrusted PR commit here is safe.
    """
    updated: list[str] = []
    for result_dir in failure_dirs:
        for subdir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
            for src_file in sorted(subdir.glob("*")):
                updated.append(
                    str(
                        Path(TEST_SUBDIR)
                        / subdir.name
                        / REGRESSION_SUBDIR
                        / src_file.name
                    )
                )

    if not updated:
        log(LogLevel.ERROR, "No regression output changes were found")
        return (Status.CANCELLED, None, {})

    patch_dir = patch_dir.resolve()

    with tempfile.TemporaryDirectory() as worktree_str:
        worktree = Path(worktree_str)
        run_git(repo_root, "worktree", "add", "--detach", str(worktree), pr_ref)
        try:
            for result_dir in failure_dirs:
                for subdir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
                    dest_dir = worktree / TEST_SUBDIR / subdir.name / REGRESSION_SUBDIR
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for src_file in sorted(subdir.glob("*")):
                        (dest_dir / src_file.name).write_bytes(src_file.read_bytes())

            run_git(worktree, "add", "--", *updated)
            run_git(
                worktree,
                "-c",
                "user.name=Github Action",
                "-c",
                "user.email=celeritas-project@users.noreply.github.com",
                "commit",
                f"--author={author}",
                "-m",
                "Update regression baselines from CI",
            )
            patch_dir.mkdir(parents=True, exist_ok=True)
            run_git(worktree, "format-patch", "-1", "HEAD", "-o", str(patch_dir))
        except Exception as e:
            log(LogLevel.ERROR, f"Failure during patch build: {e}")
            raise
        finally:
            run_git(
                repo_root, "worktree", "remove", "--force", str(worktree), check=False
            )
    updated_list = "\n".join(f"- `{f}`" for f in updated)
    out_text = f"""\
The following regression baselines differ from the recorded expected output:
{updated_list}

If these changes are expected, apply the attached patch to update them.
If not, this indicates a regression in the physics output that should be investigated before merging.

[View the failing test output in the GitHub Actions run]({actions_run_url})
"""
    return (Status.FAILURE, out_text, {"patched-files": json.dumps(updated_list)})


def process_diff(
    repo_root: Path, ref_range: str, *, actions_run_url: str
) -> tuple[Status, str | None, dict[str, str]]:
    changed_files = [
        path
        for path in run_git(repo_root, "diff", "--name-only", ref_range).splitlines()
        if PurePosixPath(path).match(f"{TEST_SUBDIR}/**/{REGRESSION_SUBDIR}/*")
    ]
    if not changed_files:
        log(LogLevel.DEBUG, "No diff result")
        return (
            Status.SUCCESS,
            "Regression tests passed and regression data matches `develop`.\n",
            {},
        )

    log(LogLevel.NOTICE, f"{len(changed_files)} files were changed")

    diff_stat = run_git(
        repo_root, "diff", "--stat=100", ref_range, "--", *changed_files
    ).strip()
    log(
        LogLevel.DEBUG,
        "Diff: " + diff_stat.splitlines()[-1],
    )

    diff_text = run_git(repo_root, "diff", ref_range, "--", *changed_files)

    truncated_note = ""
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS]
        truncated_note = (
            f"_(diff truncated to {MAX_DIFF_CHARS} characters; "
            f"see the [Actions run]({actions_run_url}) for the full diff)_\n"
        )
    comment = textwrap.dedent(f"""\
    Changes to regression baselines under `test/**/regression/*` were detected compared to `develop`.

    ```
    {diff_stat}
    ```

    [View the GitHub Actions run]({actions_run_url})
    """)
    return (Status.CHANGED, comment, {})


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing regression artifacts",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Repository root directory",
    )
    parser.add_argument(
        "--author",
        required=True,
        metavar="USER <EMAIL>",
        help="Git author for generating patch",
    )
    parser.add_argument(
        "--pr-ref",
        required=True,
        metavar="REF",
        help="local ref/sha for the (untrusted) PR head commit, fetched as inert data only",
    )
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        metavar="REF",
        help="trusted base ref to diff against (default: the checked-out base branch)",
    )
    parser.add_argument(
        "--actions-run-url",
        required=True,
        metavar="URL",
        help="GitHub Actions run URL for linked diagnostics",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Parent to write patch in case of failure",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        nargs="?",
        metavar="FILE",
        help="Path to write the comment (default stdout)",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    # Output variables: default to cancelled
    gha_output: dict[str, str] = {}
    comment: str | None = None
    status: Status = Status.CANCELLED

    artifacts_dir = Path(args.artifacts_dir)
    log(LogLevel.DEBUG, f"Artifacts dir contents for {artifacts_dir}:")
    for entry in sorted(artifacts_dir.iterdir()):
        kind = "dir" if entry.is_dir() else "file"
        log(LogLevel.DEBUG, f"  - {entry.name} ({kind})")

    if failure_dirs := find_result_dirs(args.artifacts_dir, REGRESSION_OUTPUT_ARTIFACT):
        # Diffs were generated by the build-regression run
        log(LogLevel.DEBUG, "Found regression failures")
        status, comment, gha_output = process_failures(
            failure_dirs=failure_dirs,
            repo_root=repo_root,
            author=args.author,
            pr_ref=args.pr_ref,
            patch_dir=args.patch_dir,
            actions_run_url=args.actions_run_url,
        )
    elif find_result_dirs(args.artifacts_dir, TEST_OUTPUT_ARTIFACT):
        log(LogLevel.DEBUG, "Found test output")
        # Tests were actually run (so job wasn't cancelled)
        merge_base = run_git(
            repo_root, "merge-base", args.base_ref, args.pr_ref
        ).strip()
        log(LogLevel.DEBUG, f"Comparing against merge base {merge_base}")
        status, comment, gha_output = process_diff(
            repo_root,
            f"{merge_base}..{args.pr_ref}",
            actions_run_url=args.actions_run_url,
        )
    else:
        log(LogLevel.DEBUG, "No artifacts were found: assuming cancellation")

    gha_output["status"] = str(status)

    if args.output is None:
        print(comment)
    elif comment is not None:
        log(LogLevel.DEBUG, f"Writing comment to {args.output}")
        args.output.write_text(comment)
    else:
        log(LogLevel.DEBUG, f"No comment: status is {status}")

    if (gha_filename := os.environ.get("GITHUB_OUTPUT")) is not None:
        log(LogLevel.DEBUG, f"Writing GHA output to {gha_filename!r}")
        with open(gha_filename, "a") as f:
            f.writelines(f"{k}={v}\n" for (k, v) in gha_output.items())
    elif args.output is not None:
        log(LogLevel.DEBUG, f"Writing output to {args.output!r}")
        with open(args.output, "a") as f:
            f.write(f"<!-- {gha_output!r} -->\n")
    else:
        log(LogLevel.DEBUG, "Writing output to stdout")
        print("Result:", str(gha_output))

    if status == Status.CANCELLED:
        log(
            LogLevel.WARNING,
            "Regression tests were cancelled or did not report results",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
