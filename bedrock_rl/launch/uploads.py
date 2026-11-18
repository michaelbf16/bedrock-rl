"""Private-file exclusions for explicitly uploaded training data.

Docker excludes generated datasets and checkpoints from the source context.
An explicit data mount must retain those payloads while still excluding
credentials and local tooling directories, at any depth in the upload.
"""

DATA_UPLOAD_IGNORE = (
    "**/.git", "**/.venv", "**/venv", "**/__pycache__", "**/*.py[cod]",
    "**/.pytest_cache", "**/.ruff_cache", "**/.mypy_cache", "**/.cache",
    "**/.huggingface", "**/.modal", "**/.modal.toml",
    "**/.aws", "**/.azure", "**/.ssh", "**/.config/gcloud",
    "**/.netrc", "**/.pypirc", "**/.npmrc", "**/.git-credentials",
    "**/.env", "**/.env.*", "!**/.env.example",
    "**/*.pem", "**/*.key", "**/credentials.json", "**/token.json",
    "**/*.egg-info", "**/.DS_Store",
)
