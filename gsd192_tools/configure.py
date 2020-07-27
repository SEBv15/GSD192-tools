import time as tm
import numpy as np
import zmq
from functools import reduce
from .zclient import zclient

global_default_settings = {
    "TM": False,
    "SBM": True,
    "SAUX": True,
    "SP": True,
    "SLH": True,
    "C0:C4": 21, # 0-31
    "SS0:SS1": 0, # 0-3
    "TR0:TR1": 1, # 0-3
    "SSE": False,
    "SPUR": False,
    "RT": False,
    "SL": False,
    "SB": True,
    "SBN": True,
    "M1": False,
    "M0": True,
    "SENF2": True,
    "SENF1": False,
    "RM": True,
    "PB0:PB9": 102, # 0-1023
    "gain": 0, # 0:200, 1:100, 2:50, 3:25 keV
    "shaping": 2, # 0:0.25, 1:1, 2:0.5, 3:2 usec
    "threshold": 215 # 0-1023. Aka PA0:PA9
}

channel_default_settings = {
    "ST": False,
    "SM": False,
    "SEL": True,
    "DA0:DA2": 0,
    "DP0:DP3": 0
}

def configure(gain=0, shaping=2, threshold=215, ip_addr="tcp://10.0.143.160", print_data=False, zc=None):
    """
    Configures the Germanium Detector.

    Parameters:
        gain (int): Gain configuration number 0-3. (0:200, 1:100, 2:50, 3:25 keV)
        shaping (int): The shaping time number 0-3. (0:0.25, 1:1, 2:0.5, 3:2 usec)
        threshold (int): Threshold number 0-1023.
        ip_addr (string): IP Address of the detector including protocol (Ex: tcp://10.0.143.160)
        print_data (boolean): Whether to print the configuration words or not.
        zc (object): Pass zclient instance to use instead of creating a new one.
    """
    if zc is None:
        zc = zclient(ip_addr)

    global_settings = global_default_settings.copy()
    global_settings["gain"] = gain
    global_settings["shaping"] = shaping
    global_settings["threshold"] = threshold

    if not global_settings["gain"] in range(0, 4):
        raise ValueError("Gain can only be a number 0-3. (0:200, 1:100, 2:50, 3:25 keV)")
    if not global_settings["shaping"] in range(0, 4):
        raise ValueError("Shaping can only be a number 0-3. (0:0.25, 1:1, 2:0.5, 3:2 usec)")
    if not global_settings["threshold"] in range(0, 1024):
        raise ValueError("Threshold can only be a number 0-1023.")

    channel_settings = channel_default_settings.copy()

    # DO GLOBAL SETTINGS
    mars_msw_bits = [
        global_settings["TM"] << 31,
        global_settings["SBM"] << 30,
        global_settings["SAUX"] << 29,
        global_settings["SP"] << 28,
        global_settings["SLH"] << 27,
        global_settings["gain"] << 25,
        global_settings["C0:C4"] << 20,
        global_settings["SS0:SS1"] << 18,
        global_settings["TR0:TR1"] << 16,
        global_settings["SSE"] << 15,
        global_settings["SPUR"] << 14,
        global_settings["RT"] << 13,
        global_settings["shaping"] << 11,
        global_settings["SL"] << 10,
        global_settings["SB"] << 9,
        global_settings["SBN"] << 8,
        global_settings["M1"] << 7,
        global_settings["M0"] << 6,
        global_settings["SENF2"] << 5,
        global_settings["SENF1"] << 4,
        global_settings["RM"] << 3,
        global_settings["PB0:PB9"] >> 7,
    ]

    # Bitwise And the list together
    mars_msw = reduce(lambda acc, elem : acc | elem, mars_msw_bits)

    mars_mid13 = global_settings["PB0:PB9"] << 25 | global_settings["threshold"] << 15

    if print_data:
        print("Global Settings Words: {0:032b}, {1:032b}".format(mars_msw, mars_mid13))

    # DO CHANNEL SETTINGS
    data = []

    for i in range(0, 32):
        # We are just using the same setting for every channel
        data += [
            channel_settings["ST"], 
            channel_settings["SM"], 
            0, 
            channel_settings["SEL"],
            channel_settings["DA0:DA2"] & 4,
            channel_settings["DA0:DA2"] & 2,
            channel_settings["DA0:DA2"] & 1,
            0,
            channel_settings["DP0:DP3"] & 8,
            channel_settings["DP0:DP3"] & 4,
            channel_settings["DP0:DP3"] & 2,
            channel_settings["DP0:DP3"] & 1,
        ]
        
    data.reverse()

    mars_mid = []
    for i in range(0, 12):
        word = 0
        for j in range(0, 32):
            word = word << 1 | data.pop()
        mars_mid.append(word)
        
    if print_data:
        print("MARS MID: " + ", ".join(map(lambda elem : "{0:032b}".format(elem), mars_mid)))

    # SEND TO DETECTOR
    zc.write(0,4)
    zc.write(0,0)

    zc.write(8,mars_msw)
    zc.write(0,2)
    zc.write(0,0)
    tm.sleep(0.01)

    zc.write(8,mars_mid13)
    zc.write(0,2)
    zc.write(0,0)
    tm.sleep(0.01)

    for i in range(0,12):
        zc.write(8,mars_mid[i])
        zc.write(0,2)
        zc.write(0,0)
        tm.sleep(0.01)

    zc.write(0,0x0FFF0000)
    zc.write(0,0)
    tm.sleep(0.01)

    zc.write(0,0)

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-g", "--gain", dest="gain", default=0, type=int,
                        help="Gain number 0-3. (0:200, 1:100, 2:50, 3:25 keV) (Default: 0)")
    parser.add_argument("-s", "--shaping", dest="shaping", default=2, type=int,
                        help="Shaping number 0-3. (0:0.25, 1:1, 2:0.5, 3:2 usec) (Default: 2)")
    parser.add_argument("-t", "--threshold", dest="threshold", default=215, type=int,
                        help="Threshold number 0-1023. (Default: 215)")
    parser.add_argument("-a", "--address", dest="ip_addr", default="tcp://10.0.143.160", type=str,
                        help="Address of the detector. (Default: \"tcp://10.0.143.160\")")
    args = parser.parse_args()

    if args.gain == 0 and args.shaping == 2 and args.threshold == 215:
        print("You can give gain, shaping, threshold, and address as commandline arguments. Run with -h to see options. Using default values...")

    configure(gain=args.gain, shaping=args.shaping, threshold=args.threshold, ip_addr=args.ip_addr, print_data=True)

if __name__ == "__main__":
    main()
