# GSD192
Collection of python tools for the Germanium Strip Detector

## Commandline scripts

### `gsd192-configure`

Allows simple configuration of detector like threshold, gain, and shaping time. Run with `-h` to see options. It has the exact same functionality as `gsd192_tools.configure`.

### `gsd192-monitor`

Opens a simple matplotlib graph that refreshes every 10 seconds and shows the total counts for each strip since the start of a measurement.

## Python library

### `gsd192_tools.bluesky.GSD192`

A class that can be instantiated and then used like a regular ophyd detector in bluesky.

```python
from gsd192_tools.bluesky import GSD192

det = GSD192(
    ip_addr="tcp://10.0.143.160", # Connection string for zclient
    name="GSD192", # Name to be used by bluesky. Just leave as default.
    keep_configuration=False # By default, the default gsd192_tools.configure arguments are populated to the detector at instantiation. Set True to avoid that.
)

det.configure(time=5) # Set data collection time to 5 seconds (default: 60)
det.configure(gain=2, threshold=2) # Change detector gain and threshold

from bluesky.plans import count
RE(count([det])) # Collect data for 5 seconds
```

### `gsd192_tools.configure`

Function that allows basic configuration of detector like threshold, gain, and shaping time.

```python
from gsd192_tools import configure

configure(gain=0, shaping=2, threshold=215, ip_addr="tcp://10.0.143.160", print_data=False, zc=None)
```

**Parameters:**
 - **gain (int)**: Gain configuration number 0-3. (0:200, 1:100, 2:50, 3:25 keV)
 - **shaping (int)**: The shaping time number 0-3. (0:0.25, 1:1, 2:0.5, 3:2 usec)
 - **threshold (int)**: Threshold number 0-1023.
 - **ip_addr (string)**: IP Address of the detector including protocol (Ex: tcp://10.0.143.160)
 - **print_data (boolean)**: Whether to print the configuration words or not.
 - **zc (object)**: Pass zclient instance to use instead of creating a new one.

### `gsd192_tools.zclient`

A class with some methods to make talking to the detector over ZeroMQ easier. 
