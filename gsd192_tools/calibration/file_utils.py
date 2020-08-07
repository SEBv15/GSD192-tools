import pandas as pd
import numpy as np

def loadMCA(path:str) -> np.ndarray:
    dataFile = pd.read_csv(path, sep='  ', header=None, skiprows=4, engine='python')
    return dataFile.values

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
