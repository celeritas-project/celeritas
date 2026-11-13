---
name: pull-request
description: Open a Celeritas pull request through the API or gh CLI with the required disclosure, draft status, and labels. Use whenever an agent opens or comments on a PR.
---

Open PRs from your fork's branch as a **draft**; never mark ready for review.
Valid label keywords are defined in
`.github/workflows/pull_request_target_opened.yml` (`labels` in `config`): check them
there before filing.

## Description format

```
<summary of the change>

@github-actions label geometry
@github-actions label physics

🤖 Generated with <agent harness> (<model>)
```

- The 🤖 emoji in the description makes the workflow add `ai-generated`
  (commit `Assisted-by` trailers alone give `ai-assisted`/`ai-generated`).
- Commands are one per line, alone on the line:
  `@github-actions label <label>` (zero or more allowed labels). Processed
  commands become HTML comments; unrecognized ones get a reply comment.
  There is no `task` command for PRs.
- Title: imperative, capitalized, no trailing period, no emoji.
- `gh pr create --draft --repo celeritas-project/celeritas --title ... --body-file ...`

## Every comment

Every comment or review reply on a PR or issue must disclose the harness/model
and contain a clear-text 🤖, e.g. `🤖 (Claude Code, Sonnet 5.5)`.

## Credit

`ai-generated` work is not eligible for `scripts/release/contributors.jsonl`.
