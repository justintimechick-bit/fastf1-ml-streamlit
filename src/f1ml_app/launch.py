"""``f1ml-app`` / ``python -m f1ml_app``: start the interface with Streamlit.

Extra arguments are passed to ``streamlit run`` (for example ``--server.port 8600``). Run it
from the project root so ``.streamlit/config.toml`` and the default ``data/`` paths apply, or
set ``F1ML_PROJECT_ROOT``.
"""

import sys
from pathlib import Path


def main() -> int:
    from streamlit.web import cli

    script = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(script), *sys.argv[1:]]
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
