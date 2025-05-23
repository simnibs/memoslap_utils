from ._version import __version__
from .project_settings import projects
from .run import run
from .preparation import create_cereb_surface, get_central_gm_with_mask, get_center_pos
from .simulation import run_FEMs, analyse_simus
from .reporting import placement_guide, internal_report

from simnibs import __version__
isSimNIBS4 = int(__version__[0])>3
if isSimNIBS4:
    from .write_nnav import write_nnav_files
