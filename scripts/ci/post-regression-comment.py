#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Create or update the single regression-status PR comment.

Run after ``process-regression.py`` (and, for a ``failure`` status, after the
resulting patch has been uploaded as a workflow artifact). Performs all
GitHub API interaction for this job in one execution: finds any previous
regression-bot comment on the PR and edits it, or creates a new one.
"""

import argparse
import os
import re
import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from github import Github


class RegressionStatus(StrEnum):
    failure = "failure"
    changed = "changed"
    success = "success"


BOT_MARKER = "<!-- celeritas-regression-bot -->"
STATUS_MARKER_RE = re.compile(r"<!-- celeritas-regression-status: (\w+) -->")

STATUS_STR = {
    RegressionStatus.failure: "⛔️ Regression tests failed",
    RegressionStatus.changed: "⚠️ Regression data changed",
    RegressionStatus.success: "✅ Regression tests passed",
}


def find_bot_comment(pull_request):
    for comment in pull_request.get_issue_comments():
        if comment.body.startswith(BOT_MARKER):
            return comment
    return None


def previous_status(comment) -> RegressionStatus | None:
    if comment is None:
        return None
    match = STATUS_MARKER_RE.search(comment.body)
    if match is None:
        return None
    try:
        return RegressionStatus(match.group(1))
    except ValueError:
        return None


def build_body(
    *,
    status: RegressionStatus,
    body_text: str,
    prior_status: RegressionStatus | None,
    artifact_url: str,
) -> str:
    parts = [BOT_MARKER, "# Regression test", STATUS_STR.get(status, status.value)]
    if status == RegressionStatus.success and prior_status in (
        RegressionStatus.failure,
        RegressionStatus.changed,
    ):
        parts.append("Previously reported regression issues are now resolved.")
    parts.append(body_text.strip())
    if status == RegressionStatus.failure and artifact_url:
        parts.append(
            "Download the patch to update the expected regression output:\n"
            f"[regression-update.patch]({artifact_url})\n"
            "```sh\n"
            "git am regression-update.patch\n"
            "```"
        )
    parts.append(f"<!-- celeritas-regression-status: {status.value} -->")
    return "\n\n".join(parts)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--comment-body-file", type=Path, required=True)
    parser.add_argument("--artifact-url", default="")
    args = parser.parse_args(argv)

    if args.status == "cancelled":
        # Already logged as a workflow warning by process-regression.py
        return 0

    status = RegressionStatus(args.status)

    gh = Github(os.environ["GITHUB_TOKEN"])
    github_repo = gh.get_repo(args.repo)
    pull_request = github_repo.get_pull(args.pr_number)

    existing = find_bot_comment(pull_request)
    prior_status = previous_status(existing)

    if existing is None and status == RegressionStatus.success:
        # Nothing to report and nothing to clean up
        return 0

    body = build_body(
        status=status,
        body_text=args.comment_body_file.read_text(),
        prior_status=prior_status,
        artifact_url=args.artifact_url,
    )

    if existing is not None:
        existing.edit(body=body)
    else:
        pull_request.create_issue_comment(body=body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
