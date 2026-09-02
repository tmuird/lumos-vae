"""Container types for decomposition results: a spectrum with its wavenumber
axis, and the recovered Raman spectrum, fluorophore bases and rates."""

from dataclasses import dataclass
from typing import Optional, Union, List

import numpy as np


@dataclass
class SpectralData:
    """A spectrum or stack of spectra with an associated wavenumber axis."""

    intensities: np.ndarray
    wavenumbers: Optional[np.ndarray] = None
    label: Optional[Union[str, List[str], np.ndarray]] = None
    time_values: Optional[np.ndarray] = None

    @property
    def data(self) -> np.ndarray:
        return self.intensities


@dataclass
class DecompositionResult:
    """Recovered Raman spectrum, fluorophore bases, abundances and decay rates."""

    raman: SpectralData
    fluorophore_spectra: SpectralData
    abundances: Optional[np.ndarray] = None
    rates: Optional[np.ndarray] = None
    frame_duration: Optional[float] = None
