"""
Bias CLI Module
===============

Command-line interface for the Bias library.

Usage
-----
$ bias steer "professional writing" --intensity 2.0
$ bias generate "Write an email:" --concept "formal"
$ bias discover "technical language"
$ bias interactive
"""

from bias.cli.main import app, main

__all__ = ["app", "main"]

