import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parent.parent      # minemag/backend
_REPO = _BACKEND.parent.parent                                  # repository root (magcore/)
for p in (str(_BACKEND), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES = _BACKEND / "tests" / "fixtures"
R3_FIXTURE = FIXTURES / "20260425_001316_MD-R3_#0115_noff_sf5_mm3(ROBOT1)_2Hz_tracks10-11.asc"
