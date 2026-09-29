#!/usr/bin/env python
"""Run the predeclared raw-identity FuXi-to-IMD neural ablation.

This thin entry point fixes the one-factor experimental contract and delegates
the shared feature, training, validation, and packaging implementation to
``fuxi_imd_attention_climatology``.  It intentionally does not expose a switch
back to the fitted log-bias training anchor.
"""

from __future__ import annotations

import sys

import fuxi_imd_attention_climatology as experiment


FIXED_ARGUMENTS = (
    "--all-weeks",
    "--full-fuxi-context",
    "--training-anchor",
    "raw_fuxi",
)
FORBIDDEN_ARGUMENTS = {
    "--all-weeks",
    "--full-fuxi-context",
    "--large-model",
    "--regularized-large",
    "--training-anchor",
}


def main() -> None:
    supplied = {
        value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")
    }
    conflicts = sorted(supplied & FORBIDDEN_ARGUMENTS)
    if conflicts:
        raise SystemExit(
            "the no-log-bias contract fixes these arguments; remove: "
            + ", ".join(conflicts)
        )
    sys.argv[1:1] = list(FIXED_ARGUMENTS)
    experiment.main()


if __name__ == "__main__":
    main()
