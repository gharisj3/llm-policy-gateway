"""Install the pinned spaCy pipeline selected by PRESIDIO_NLP_MODEL."""

import os
import subprocess
import sys

MODEL_VERSION = "3.8.0"
SUPPORTED_MODELS = {"en_core_web_sm", "en_core_web_lg"}


def main() -> int:
    model = os.getenv("PRESIDIO_NLP_MODEL", "en_core_web_lg")
    if model not in SUPPORTED_MODELS:
        print(f"Unsupported PRESIDIO_NLP_MODEL: {model}", file=sys.stderr)
        return 1
    wheel = f"{model}-{MODEL_VERSION}-py3-none-any.whl"
    url = (
        "https://github.com/explosion/spacy-models/releases/download/"
        f"{model}-{MODEL_VERSION}/{wheel}"
    )
    return subprocess.call([sys.executable, "-m", "pip", "install", "--no-deps", url])


if __name__ == "__main__":
    raise SystemExit(main())
