# Commandline Scripts

## gsd192-configure

*Commandline version of the [`gsd192_tools.configure`](https://github.com/SEBv15/GSD192-tools/blob/master/gsd192_tools/configure.py) function*

Allows simple configuration of detector like threshold, gain, and shaping time. To configure more parameters or define different settings for each ASIC, use the mars_config GUI.

#### Example

```
$ gsd192-configure -g 0 -s 2 -t 215 -a "tcp://10.0.143.160"
```

## gsd192-monitor

Watches incoming data from the detector and displays a live graph that refreshes every couple seconds.

**This script is written purely in python and the performance was only tested on calibration sources. It might be too slow if there are more events per second.**

#### Example

```
$ gsd192-monitor -a "tcp://10.0.143.160"
```

## gsd192-time-mca

Collects data for a given number of seconds and saves it to an `.mca` file.

#### Example

```
$ gsd192-time-mca 30 30sec_count.mca "tcp://10.0.143.160"
```

*The last argument containing the IP address is optional*