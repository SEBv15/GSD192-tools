import numpy as np
import pandas as pd
import math

from lmfit import Minimizer, Parameters, report_fit, Parameter, minimize
from lmfit.models import GaussianModel, Model, RectangleModel, LorentzianModel, LinearModel

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import spatial
from scipy.interpolate import interp1d

from functools import reduce

import time
import json
import copy

from dtw import dtw, warp, rabinerJuangStepPattern

def loadMCA(path):
    dataFile = pd.read_csv(path, sep='  ', header=None, skiprows=4, engine='python')
    return dataFile.values

class Pixel:
    def __init__(self, pixelData, peakEnergies=None, pixelNumber=None, prominenceScaleFactor=0.09):
        self.data = pixelData
        self.number = pixelNumber
        self.prominenceScaleFactor = prominenceScaleFactor
        self.peakEnergies = peakEnergies
        self.transformations = []
        self.peaks = None
        self.fitParams = None
        self.fitModel = None
        self.fitPeaks = None
        self.additionalFitPeaks = []

    def detectPeaks(self, showPlot=False):
        """
        Find the peaks in this pixel. Used to estimate where the gaussians should go for the curve fit.

        :return: Array of xy coordinate tuples representing peak locations (sorted left to right)
        """
        scale = sum(self.data)
        smoothed = savgol_filter(self.data, 5, 3)
        peaks = find_peaks(smoothed, prominence=(scale**0.5)*self.prominenceScaleFactor, distance=10)
        self.peaks = np.asarray(list(map(lambda x : (int(x), int(self.data[x])), peaks[0])))

        if showPlot:
            plt.plot(smoothed, label="Data")
            plt.plot(*zip(*self.peaks), marker='x', label="Detected Peaks", linestyle = 'None')
            plt.legend()
            plt.show()

        return self.peaks

    def detectPeaksLog(self, showPlot=False, prominence=0.5, minHeight=1.5):
        """
        Find the peaks in this pixel using the logged data. Used to estimate where the gaussians should go for the curve fit.

        :return: Array of xy coordinate tuples representing peak locations (sorted left to right)
        """
        np.seterr(divide='ignore')
        smoothed = np.maximum(0, np.log10(self.data))
        smoothed = savgol_filter(smoothed, 5, 3)
        peaks = find_peaks(smoothed, prominence=prominence, distance=10, height=minHeight)
        logPeaks = np.asarray(list(map(lambda x : (int(x), int(smoothed[x])), peaks[0])))
        self.peaks = np.asarray(list(map(lambda x : (int(x), int(self.data[x])), peaks[0])))

        if showPlot:
            plt.plot(smoothed, label="Logged and smoothed data")
            plt.plot(*zip(*logPeaks), marker='x', label="Detected Peaks", linestyle = 'None')
            plt.legend()
            plt.show()

        return self.peaks

    def transformationTo(self, otherPixel, showPlot=False, dtwDerivative=True, smoothed=True):
        if otherPixel.number != None:
            matching = list(filter(lambda trans : trans[0] == otherPixel.number, self.transformations))
            if len(matching) > 0:
                return matching[0][1]

        if type(otherPixel) != Pixel:
            raise ValueError("Expected argument to be a pixel")

        deriv = np.gradient(self.data, axis=0)
        
        if dtwDerivative:
            selfData = np.gradient(self.data, axis=0)
            otherData = np.gradient(otherPixel.data, axis=0)
        else:
            selfData = self.data
            otherData = otherPixel.data

        alignment = dtw(
            selfData,
            otherData,
            keep_internals=True, 
            open_begin=True, # pixels have different thresholds where they start detecting data
            step_pattern=rabinerJuangStepPattern(6, "c", smoothed=smoothed),
            )

        t = warp(alignment, index_reference=False)

        if showPlot:
            alignment.plot(type="twoway",offset=-2)
            plt.figure("DTW result"+(" from {} to {}".format(self.number, otherPixel.number) if self.number is not None and otherPixel.number is not None else ""))
            plt.plot(otherPixel.data, label="Alignment target"+(" ({})".format(otherPixel.number) if otherPixel.number != None else ""))
            plt.plot(self.data[t], label="This pixel"+(" ({})".format(self.number) if self.number != None else ""))
            plt.plot(deriv[t], label="Derivative of data for pixel")
            plt.legend()
            plt.show()

        if otherPixel.number != None:
            self.transformations.append((otherPixel.number, t))

        return t

    def setPeakEnergies(self, peakEnergies):
        self.peakEnergies = peakEnergies

    def _getTransformFunction(self, transformation):
        return interp1d(transformation, list(range(len(transformation))))

    def usePeakEnergiesFrom(self, otherPixel, showPlot=False):
        if type(otherPixel) != Pixel:
            raise ValueError("Expected argument to be a pixel")
        if otherPixel.peakEnergies == None:
            raise Exception("Other pixel doesn't have peak energies set")

        transformation = otherPixel.transformationTo(self, showPlot=showPlot)

        f = interp1d(transformation, list(range(len(transformation))))

        self.peakEnergies = []
        for energy in otherPixel.peakEnergies:
            self.peakEnergies.append((f(energy[0]), energy[1]))

    def getLabeledPeaks(self, additionalPeaks=[], ignoreFitPeaks=False, logPeakDetectProminence=0.5, logPeakDetectMinHeight=1.5):
        if not self.peakEnergies:
            raise Exception("No peak energies set")

        if self.peaks is None:
            self.detectPeaksLog(prominence=logPeakDetectProminence, minHeight=logPeakDetectMinHeight)

        peaks = self.peaks

        if self.fitPeaks is not None and not ignoreFitPeaks:
            peaks = self.fitPeaks

        if len(additionalPeaks) > 0:
            peaks = sorted(np.concatenate((additionalPeaks, peaks)).tolist(), key=lambda peak : peak[0])

        out = []

        for energy in self.peakEnergies:
            peakD = list(map(lambda peak : abs(peak[0]-energy[0]), peaks))
            peak = peaks[peakD.index(min(peakD))]
            out.append([peak[0], peak[1], energy[1]])

        out = sorted(out, key=lambda peak : peak[0])

        return out

    def getEnergyXValues(self, fitOrder=3, showPlot=False):
        peaks = self.getLabeledPeaks()

        if len(peaks) < 2:
            raise Exception("need at least two peaks with known energies")

        if fitOrder > len(peaks):
            fitOrder = len(peaks)

        def transformN(number, pars):
            return sum(((number)**n) * pars["p"+str(n)] for n in range(len(pars)))

        def distanceToMinimize(pars, peaks):
            return np.array(list(map(lambda peak : abs(peak[2]-transformN(peak[0], pars)), peaks)))

        params = Parameters()
        params["p0"] = Parameter("p0", value=0.0)
        params["p1"] = Parameter("p1", value=1.0)
        for n in range(2, fitOrder):
            params["p"+str(n)] = Parameter("p"+str(n), value=0.0)

        out = minimize(distanceToMinimize, params, args=(peaks,))

        if showPlot:
            plt.plot(self.data)
            plt.plot(*zip(*peaks), marker="x", linestyle = 'None')
            plt.show()

        pars = out.params.valuesdict()

        return list(map(lambda p : transformN(p, pars), list(range(len(self.data)))))

    def curveFit(self, reuseAdditionalPeaksFrom=None, reuseModelFrom=None, additionalPeaks=[], showPlot=False):
        fitData = self.data.copy()

        if reuseAdditionalPeaksFrom is not None:
            additionalPeaks = []
            transformation = reuseAdditionalPeaksFrom.transformationTo(self)
            f = self._getTransformFunction(transformation)
            for peak in reuseAdditionalPeaksFrom.additionalFitPeaks:
                additionalPeaks.append((int(f(peak[0])), peak[1]))
        elif len(additionalPeaks) > 0:
            self.additionalFitPeaks = additionalPeaks

        labeledPeaks = self.getLabeledPeaks(additionalPeaks=additionalPeaks, ignoreFitPeaks=True)

        if len(additionalPeaks) > 0:
            peaks = self.peaks.copy()
            i = 0
            # Remove possible duplicates that can occur when an additionalPeak is actually detected already
            for p in peaks:
                if min(np.abs(np.asarray(list(zip(*additionalPeaks))[0])-p[0])) < 10:
                    peaks = np.delete(peaks, i, axis=0)
                else:
                    i += 1
            allPeaks = sorted(np.concatenate((additionalPeaks, peaks)).tolist(), key=lambda peak : peak[0])
        else:
            allPeaks = self.peaks.tolist()

        cutoff = 0

        # Get rid of unnecessary peaks on the left if possible
        if len(labeledPeaks) != 0:
            firstLabeledPeakIndex = allPeaks.index(labeledPeaks[0][0:2])
            if firstLabeledPeakIndex >= 2:
                cutoff = int(allPeaks[firstLabeledPeakIndex-1][0])-1

        self.fitCutoff = cutoff
        fitData = fitData[cutoff:]

        if reuseModelFrom is None or reuseModelFrom.fitModel is None:
            models = []
            params = Parameters()

            # Add all detected peaks
            for i in range(len(allPeaks)):
                peak = allPeaks[i]
                if peak[0] < cutoff:
                    continue
                prefix = "p{}_".format(i)
                models.append(GaussianModel(prefix=prefix))
                params.add(prefix+"center", value=peak[0]-cutoff, min=peak[0]-cutoff-5, max=peak[0]-cutoff+5)
                params.add(prefix+"sigma", value=5)
                params.add(prefix+"amplitude", value=peak[1]*5*math.sqrt(2*math.pi))
                params.add(prefix+"height", value=peak[1])

            # Add boxes
            for i in range(len(allPeaks)):
                vary = (i == len(allPeaks) - 1)
                numP = len(allPeaks)
                peakA = allPeaks[min(i%numP, (i+1)%numP)]
                peakB = allPeaks[max(i%numP, (i+1)%numP)]

                if peakA[0] < cutoff:
                    continue

                prefix = "b{}_".format(i)
                models.append(RectangleModel(prefix=prefix, form="arctan"))
                params.add(prefix+"amplitude", value=min(fitData[peakA[0]-cutoff:peakB[0]-cutoff]), vary=vary)
                params.add(prefix+"center1", value=peakA[0]-cutoff-5, vary=vary)
                params.add(prefix+"center2", value=peakB[0]-cutoff-5, vary=vary)
                params.add(prefix+"sigma1", value=10, vary=vary)
                params.add(prefix+"sigma2", value=10, vary=vary)
            
            self.fitParams = params
            self.fitModel = reduce(lambda acc, m : acc + m, models)
        else:
            # Reuse model from another fit and just adjust center parameters. This doesn't work well if peaks are only exist in some graphs but not in others
            self.fitModel = reuseModelFrom.fitModel
            self.fitParams = Parameters()

            transformation = reuseModelFrom.transformationTo(self)
            f = self._getTransformFunction(transformation)

            for param, value in reuseModelFrom.fitParams.items():
                p = Parameter(param, value=value.value, vary=value.vary)
                if "center" in param:
                    p.set(value=f(value.value+reuseModelFrom.fitCutoff)-cutoff)
                self.fitParams[param] = p

        out = self.fitModel.fit(fitData, self.fitParams, x=np.array([*range(len(fitData))]))

        self.fitPeaks = []
        for key, value in out.params.valuesdict().items():
            if key.startswith('p') or key.startswith('m'):
                if "center" in key:
                    self.fitPeaks.append((value+cutoff, self.data[int(value+cutoff)]))

        if showPlot:
            plt.plot(self.data, label="Data")
            plt.plot(np.concatenate(([0]*cutoff, fitData)), label="Data without unnecessary peaks")
            plt.plot(np.concatenate(([0]*cutoff, self.fitModel.eval(self.fitParams, x=np.asarray(list(range(len(fitData))))))), label="Initial Fit Parameters")
            plt.plot(np.concatenate(([0]*cutoff, fitData + out.residual)), label="Curve Fit")
            plt.plot(*zip(*self.fitPeaks), marker="x", linestyle='None', label="Peaks from curve fit")
            plt.legend()
            plt.show()

        self.fitParams = out.params

        return np.concatenate(([0]*cutoff, fitData + out.residual))
    
    def alignWith(self, otherPixel, showPlot=False):
        """
        This method can be useful to get some data in line quickly without keeping track of the transformations used to do it.
        """
        t = self.transformationTo(otherPixel, showPlot=showPlot)
        
        if self.peaks:
            print("It seems like detectPeaks was already run. Shame since they are not correct anymore and are being deleted now")
            self.peaks = None

        self.data = self.data[t]

def displayPixels(data, xmin, xmax, xlabel='Energy (keV)'):
    plt.imshow(data, cmap='hot', interpolation='nearest', aspect=(xmax-xmin)/float(len(data))/2, extent=[xmin, xmax, len(data), 0])
    plt.colorbar(spacing='uniform', fraction=0.01, pad=0.01, orientation='vertical')
    plt.title('Spectrum from each Strip')
    plt.xlabel(xlabel)
    plt.ylabel('Strip Address')
    plt.show()

def toCalibrationFile(pixels, name, units, isOrdered=False, sigfix=5):
    """
    Format an array of pixels to a calibration file containing x values for all points in the corresponding mca file.

    :param pixels: A list of pixel class instances
    :param name: The name of the data
    :param units: Energy used for calibration
    :param isOrdered: If the pixels list is already in ascending order by pixel number, set this to true
    :param sigfix: The number of significant figures to round the calculated energy values to
    :returns: A string to be saved to a .cal file
    """
    curveFitted = True
    numPixels = pixels[-1].number+1
    for pix in pixels:
        if pix.fitPeaks is None:
            curveFitted = False

        if pix.number is None and not isOrdered:
            raise ValueError("Pixels need to know their pixel number or isOrdered needs to be true")

        numPixels = max(numPixels, pix.number+1)

    lines = [""]*numPixels

    for i in range(len(pixels)):
        xdata = pixels[i].getEnergyXValues()
        lines[i if isOrdered else pixels[i].number] = "  ".join(map(lambda x : (('%.'+str(sigfix)+'g') % x), xdata))

    headers = []
    headers.append("name: {}".format(name))
    headers.append("type: CAL")
    headers.append("pixels: {}".format(numPixels))
    headers.append("channels: {}".format(len(pixels[0].data)))
    headers.append("units: {}".format(units))
    headers.append("intensity_calibration: ")
    headers.append("curve_fitted: "+("true" if curveFitted else "false"))

    return "\t#"+"\n\t#".join(headers)+"\n"+"\n".join(lines)+"\n"

def parseCalibrationFile(data, peakFile=False):
    lines = data.split("\n")
    i = 0
    out = {}
    while (lines[i].startswith("\t#")):
        out[lines[i].replace("\t#","").split(":")[0]] = lines[i].split(":")[1].strip()
        i += 1

    if "intensity_calibration" in out.keys():
        if out["intensity_calibration"] == "":
            out["intensity_calibration"] = None
        else:
            out["intensity_calibration"] = list(map(lambda x : float(x.strip()), out["intensity_calibration"].split(",")))

    data = []
    for strip in lines[i:]:
        if strip.strip() == "":
            data.append(None)
        else:
            data.append(list(map(lambda x : float(x.strip()) if not peakFile else (float(x.split(":")[0].strip()), float(x.split(":")[1].strip())), strip.split("  "))))

    out["data"] = data
    return out

def toPeakFile(pixels, name, units, isOrdered=False, xDecimals=3):
    """
    Format an array of pixels to a calibration file containing x values with known energies for all strips. This can be used to fit a custom calibration function to apply to data.

    :param pixels: A list of pixel class instances
    :param name: The name of the data
    :param units: Energy used for calibration
    :param isOrdered: If the pixels list is already in ascending order by pixel number, set this to true
    :param xDecimals: The number of decimal points to include for the x location of the energy
    :returns: A string to be saved to a .calp file
    """
    curveFitted = True
    numPixels = pixels[-1].number+1
    for pix in pixels:
        if pix.fitPeaks is None:
            curveFitted = False

        if pix.number is None and not isOrdered:
            raise ValueError("Pixels need to know their pixel number or isOrdered needs to be true")

        numPixels = max(numPixels, pix.number+1)

    lines = [""]*numPixels

    for i in range(len(pixels)):
        peaks = pixels[i].getLabeledPeaks()
        lines[i if isOrdered else pixels[i].number] = "  ".join(map(lambda peak : str(round(peak[0], xDecimals))+":"+str(peak[2]), peaks))

    headers = []
    headers.append("name: {}".format(name))
    headers.append("type: CALP")
    headers.append("pixels: {}".format(numPixels))
    headers.append("channels: {}".format(len(pixels[0].data)))
    headers.append("units: {}".format(units))
    headers.append("intensity_calibration: ")
    headers.append("curve_fitted: "+("true" if curveFitted else "false"))

    return "\t#"+"\n\t#".join(headers)+"\n"+"\n".join(lines)+"\n"

def parsePeakFile(data):
    return parseCalibrationFile(data, peakFile=True)
