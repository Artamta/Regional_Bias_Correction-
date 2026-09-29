"""Use plotting packages from the CPU environment after HPC packages resolve."""

import sys


FUXI_SITE_PACKAGES = "/home/raj.ayush/.conda/envs/fuxi/lib/python3.10/site-packages"
if FUXI_SITE_PACKAGES not in sys.path:
    sys.path.append(FUXI_SITE_PACKAGES)
