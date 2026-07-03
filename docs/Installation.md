# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Do not commit `.venv/`, `.cache/`, Hugging Face model weights, or `.env` files.

