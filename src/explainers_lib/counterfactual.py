from dataclasses import dataclass
from typing import TypeAlias
import numpy as np

ClassLabel: TypeAlias = int


# For now not sure whether it should be in a separate directory
@dataclass
class Counterfactual:
    """This is a helper class"""

    original_data: np.ndarray # might be useful as well
    changed_data: np.ndarray
    original_class: ClassLabel
    target_class: ClassLabel
