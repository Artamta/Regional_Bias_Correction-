# Contributing

This repository contains two related but scientifically independent research
workstreams. Keep changes scoped to one project unless a repository-wide change
is explicitly required.

## Development workflow

1. Read the target project's README and scientific contract.
2. Create a focused branch and keep generated artifacts out of Git.
3. Add or update synthetic tests for changes to alignment, split boundaries,
   metrics, manifests, or artifact handling.
4. Run the relevant project's cache-free test command.
5. Describe scientific impact, data-contract changes, and verification in the
   pull request.

Python code uses four-space indentation, type hints, concise docstrings,
`snake_case` functions and variables, `PascalCase` classes, and
`UPPER_SNAKE_CASE` constants. Follow nearby code where no automated formatter
is configured.

## Pull request checklist

- [ ] The change is scoped to the correct project.
- [ ] Training, validation, retrospective, and sealed-test boundaries remain
      intact.
- [ ] New behavior has focused synthetic or contract tests.
- [ ] Relevant tests pass without writing caches into the repository.
- [ ] No credentials, raw data, checkpoints, logs, or generated run artifacts
      are included.
- [ ] Scientific claims are linked to a manifest or reproducible source
      artifact and are labelled with the correct evidence role.

Do not rewrite or selectively prune completed provenance directories. If an
experiment is superseded or withdrawn, preserve that status explicitly in the
documentation.
