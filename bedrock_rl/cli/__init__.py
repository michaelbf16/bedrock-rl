"""The `bedrock` command.

One front door over the shell launchers in bin/, the python utilities
beside them, and the `python -m` module entry points, grouped by what a
user is trying to do rather than by which script exists.

This file imports nothing on purpose. The console and the theme are
bedrock_rl/cli/console.py, and a trainer that wants them should not
pay for argparse and the whole command table to print one line. The
entry point in pyproject.toml names bedrock_rl.cli.main:main directly
for the same reason.
"""
