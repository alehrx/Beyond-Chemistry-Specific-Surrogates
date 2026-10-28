"""
srcs_EM - the DIMENSIONLESS MIONet surrogate: deployment package.

The operator, its chemistry transform and deployment, plus the electrolyte (SPMe), whole-cell and
multi-slice extensions. From the repository root:

    from srcs_EM import Surrogate, Chemistry

    sur  = Surrogate.from_run("runs/v2_nondim/mionet_77k_s42")
    chem = Chemistry.from_pybamm("Mohtat2020", "Negative")
    x1   = sur.step(x0, I, dt, chem, T)
"""
from .chemistry import Chemistry, T_REF
from .mionet_nd import MIONetND
from .surrogate import Surrogate
from .electrolyte import Electrolyte
from .voltage import CellVoltage
from .cell import Cell, CellState
from .multislice import MultiSlice, SliceState

__all__ = ["Chemistry", "T_REF", "MIONetND", "Surrogate", "Electrolyte", "CellVoltage", "Cell", "CellState", "MultiSlice", "SliceState"]
