import numpy as np
from typing import List, Tuple, Any

from scipy.signal import find_peaks, savgol_filter
from dtw import dtw, warp, rabinerJuangStepPattern
from lmfit import Parameters, Parameter
from lmfit.models import GaussianModel, PolynomialModel

class Strip:
    """
    Represents a single strip/pixel of a strip detector and provides functions to calibrate the energy spectrum with other Strip instances
    """
    def __init__(self, data:np.ndarray, number:int, energies:List[Tuple[int, float]] = None):
        """
        Parameters:
            data (np.ndarray[int]): The energy spectrum for the strip
            number (int): The index of the strip
            energies (List[Tuple[int, float]]): The energies corresponding with peaks in the data (this should only be given for one strip)
        """
        self.data = data
        self.number = number
        self.energies = None
        self.refined_energies = None
        self.polynomial_coefficients = None
        self._peaks = None
        if energies:
            self.set_energies(energies)

        self._peak_prominence = 0.5
        self._peak_height = 1.5

    def set_energies(self, energies:List[Tuple[int, float]], ignore_error:bool = False) -> List[Tuple[int, float]]:
        """
        Set the known energies for peaks in the spectrum

        Parameters:
            energies (List[Tuple[int, float]]): The known energies as (x, energy) tuples
            ingore_error (bool): Function will throw an error if two energies seem to be for the same peak (most likely caused by insensitive peak detection)

        Returns:
            (List[Tuple[int, float]]): The same energies but with x values corresponding to peaks found with find_peaks
        """
        new_x = []
        for e in energies:
            x = min(self.peaks, key=lambda p : abs(p[0] - e[0]))[0]
            if x in new_x and not ignore_error:
                other = energies[new_x.index(x)]
                print(self.number)
                raise Exception("Energy ({}, {}) got matched to the same peak as ({}, {}) at {}. Changing peak_prominence or peak_height might help.".format(e[0], e[1], other[0], other[1], x))
            new_x.append(x)
        
        self.energies = [(new_x[i], energies[i][1]) for i in range(len(energies))]
        return self.energies

    @property
    def peak_prominence(self):
        """Minimum prominence of a peak for find_peaks (log scale)"""
        return self._peak_prominence

    @peak_prominence.setter
    def peak_prominence(self, value:float):
        self._peaks = None # Force recalculation
        self._peak_prominence = value

    @property
    def peak_height(self):
        """Minimum height of a peak for find_peaks (log scale)"""
        return self._peak_height

    @peak_height.setter
    def peak_height(self, value:float):
        self._peaks = None # Force recalculation
        self._peak_height = value

    @property
    def peaks(self):
        if not self._peaks:
            self._find_peaks()
        return self._peaks

    def _find_peaks(self) -> List[Tuple[int, Any]]:
        """
        Detect peaks on log scaled, smoothed data.

        Returns:
            (List[Tuple]): List of peaks represented as (x, y) tuples
        """
        logged = np.maximum(0, np.log10(self.data, where=(self.data != 0)))
        smoothed = savgol_filter(logged, 5, 3)
        res = find_peaks(smoothed, prominence=self.peak_prominence, distance=10, height=self.peak_height)
        self._peaks = [(x, self.data[x]) for x in res[0]]
        return self._peaks

    def warp_from(self, base_strip:'Strip') -> np.ndarray:
        """
        Find the dynamic time warp between strips

        Parameters:
            base_strip (Strip): Another strip
        """ 
        alignment = dtw(
            base_strip.data,
            self.data,
            keep_internals=True, 
            open_begin=True, # pixels have different thresholds where they start detecting data
            open_end=True, # In case some peak is cut off
            step_pattern=rabinerJuangStepPattern(6, "c"),
        )

        transformation = warp(alignment, index_reference=True)
        return transformation

    def energies_from(self, base_strip:'Strip', ignore_error = False) -> List[Tuple[int, Any]]:
        """
        Find the dynamic time warp between strips and copy the adjusted peak energies from the other one

        Parameters:
            base_strip (Strip): A strip with known energies
        """
        if not base_strip.energies:
            raise ValueError("base_strip doesn't have any known energies")

        transformation = self.warp_from(base_strip)

        w_energies = [(transformation[e[0]], e[1]) for e in base_strip.energies]

        return self.set_energies(w_energies, ignore_error), w_energies

    def refine_energies(self, frame_width:int = 20):
        """
        Refine energy locations using curve fitting
        """
        model = GaussianModel()
        refined = []
        for energy in self.energies:
            domain = (int(max(energy[0]-frame_width/2, 0)), int(min(energy[0]+frame_width/2, self.data.shape[0]-1)))
            frame = self.data[domain[0]:domain[1]]
            pars = model.guess(frame, x=np.arange(0, 20))
            out = model.fit(frame, pars, x=np.arange(0, 20))
            refined.append((out.params["center"].value + domain[0], energy[1]))
        
        self.refined_energies = refined
        self.polynomial_coefficients = None
        return refined

    def fit_polynomial(self, max_degree:int = 7):
        """
        Fit the highest degree polynomial possible with the available energy data. (0 degree is actually 1st degree with 0 shift)

        Returns:
            (List[Float]): The polynomial coefficients from highest degree to the constant term 
        """
        energies = self.refined_energies if self.refined_energies else self.energies
        if not energies:
            raise ValueError("No known energies found for strip {}".format(self.number))

        degree = min(len(energies) - 1, max_degree) if max_degree else len(energies) - 1

        model = PolynomialModel(max(degree, 1))
        x, y = [*zip(*energies)]
        pars = model.guess(y, x=x)
        if degree == 0:
            pars["c0"] = Parameter("c0", value=0, vary=False)

        out = model.fit(y, pars, x=x)
        self.polynomial_coefficients = list(reversed([p[1].value for p in out.params.items()]))
        return self.polynomial_coefficients

    def calibrated_x(self):
        """
        Get x values for the data that correspond to actual energy values

        Returns:
            (np.ndarray): 1D array of x values for the data
        """
        if not self.polynomial_coefficients:
            self.fit_polynomial()

        return np.polyval(self.polynomial_coefficients, np.arange(0, self.data.shape[0]))
