from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from experiment.vis.result_analysis import main
else:
    from .result_analysis import main


if __name__ == "__main__":
    main()
