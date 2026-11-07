"""Installed version of the package, recorded with every set of settings."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("morphobricks-normalization")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0.0.0+unknown"
