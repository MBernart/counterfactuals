from dataclasses import dataclass
from typing import TypeAlias, Any
from numpy.typing import NDArray

ClassLabel: TypeAlias = int


# For now not sure whether it should be in a separate directory
@dataclass
class Counterfactual:
    """This is a helper class"""

    original_data: NDArray[Any] # might be useful as well
    changed_data: NDArray[Any]
    original_class: ClassLabel
    target_class: ClassLabel
