from dataclasses import dataclass
from typing import TypeAlias, Any
from numpy.typing import NDArray
import numpy as np

ClassLabel: TypeAlias = int


# For now not sure whether it should be in a separate directory
@dataclass
class Counterfactual:
    """This is a helper class"""

    data: np.ndarray
    original_class: ClassLabel
    target_class: ClassLabel
