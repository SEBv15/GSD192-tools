import pandas as pd
import numpy as np

from typing import Union, List

def loadMCA(path:str, num_header_rows:int = 4) -> np.ndarray:
    """
    Load an `.mca` file into a numpy 2D array

    Parameters:
        path (str): The path to the file
        num_header_rows (int): Number of rows to ignore

    Returns:
        (np.ndarray): The data
    """
    dataFile = pd.read_csv(path, sep='  ', header=None, skiprows=num_header_rows, engine='python')
    return dataFile.values

def toCalibrationFile(strips:'Union[Strips, List[Strip]]', name:str, units:str, sigfix:int=5):
    """
    Format an array of strips to a calibration file containing x values for all points in the corresponding mca file.
    :param strips: An instance of the Strips class or a list of Strip instances
    :param name: The name of the data
    :param units: Energy used for calibration
    :param sigfix: The number of significant figures to round the calculated energy values to
    :returns: A string to be saved to a .cal file
    """
    curveFitted = True
    numPixels = 0
    for strip in strips:
        if strip.refined_energies is None:
            curveFitted = False

        numPixels = max(numPixels, strip.number+1)

    lines = [""]*numPixels

    for i in range(len(strips)):
        xdata = strips[i].calibrated_x()
        lines[strips[i].number] = "  ".join(map(lambda x : (('%.'+str(sigfix)+'g') % x), xdata))

    headers = []
    headers.append("name: {}".format(name))
    headers.append("type: CAL")
    headers.append("pixels: {}".format(numPixels))
    headers.append("channels: {}".format(strips[0].data.shape[0]))
    headers.append("units: {}".format(units))
    headers.append("intensity_calibration: ")
    headers.append("curve_fitted: "+("true" if curveFitted else "false"))

    return "\t#"+"\n\t#".join(headers)+"\n"+"\n".join(lines)

def toPeakFile(strips:'Union[Strips, List[Strip]]', name:str, units:str, x_decimals:int=3):
    """
    Format an array of strips to a calibration file containing x values with known energies for all strips. This can be used to fit a custom calibration function to apply to data.
    :param strips: An instance of the Strips class or a list of Strip instances
    :param name: The name of the data
    :param units: Energy used for calibration
    :param x_decimals: The number of decimal points to include for the x location of the energy
    :returns: A string to be saved to a .calp file
    """
    curveFitted = True
    numPixels = 0
    for strip in strips:
        if strip.refined_energies is None:
            curveFitted = False

        numPixels = max(numPixels, strip.number+1)

    lines = [""]*numPixels

    for i in range(len(strips)):
        peaks = strips[i].refined_energies if strips[i].refined_energies else strips[i].energies
        lines[strips[i].number] = "  ".join(map(lambda peak : str(round(peak[0], x_decimals))+":"+str(peak[1]), peaks))

    headers = []
    headers.append("name: {}".format(name))
    headers.append("type: CALP")
    headers.append("pixels: {}".format(numPixels))
    headers.append("channels: {}".format(strips[0].data.shape[0]))
    headers.append("units: {}".format(units))
    headers.append("intensity_calibration: ")
    headers.append("curve_fitted: "+("true" if curveFitted else "false"))

    return "\t#"+"\n\t#".join(headers)+"\n"+"\n".join(lines)

def parseCalibrationFile(data:str, peak_file:bool=False):
    """
    Parse a `.cal` or `.calp` file (the output of either `toCalibrationFile` or `toPeakFile`)

    Parameters:
        data (str): The contents of the file as string
        peak_file (bool): Whether it is a `.calp` file or not
    
    Returns:
        (dict): The data
    """
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
            data.append(list(map(lambda x : float(x.strip()) if not peak_file else (float(x.split(":")[0].strip()), float(x.split(":")[1].strip())), strip.split("  "))))

    out["data"] = data
    return out
