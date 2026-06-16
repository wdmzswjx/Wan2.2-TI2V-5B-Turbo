"""Project utility package.

This file makes the repository's top-level ``utils`` directory an explicit
Python package.  Without it, Python can resolve ``import utils`` to an installed
third-party package named ``utils`` before considering this namespace directory,
which breaks imports such as ``utils.wan_wrapper`` when launching training from
shell scripts.
"""
