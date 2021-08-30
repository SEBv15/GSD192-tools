from collections import OrderedDict
import numpy as np
import time as tm
from typing import Dict, List, Any, TypeVar, Tuple
import threading
import zmq
from .zclient import zclient
from .configure import configure

class Status:
    """
    Every data collection will have its own status object
    """
    def __init__(self, done=False, success=True):
        """You shouldn't have to initiate this class"""

        self.done = done
        """Set to 'True' when the data collection is finished"""
        self.success = success
        """Whether the data collection was successful"""
        self._finished_cb = None
        
        self._watchers = []
    
    def watch(self, func):
        """
        Register callback function that will receive updates about data collection progress.
        Function is expected to accept the keyword arguments: name, current, initial, target, unit, precision, fraction, time_elapsed, time_remaining

        :param func: The callback function
        """
        self._watchers.append(func)

    def _notify(self, **kwargs):
        for watcher in self._watchers:
            threading.Thread(target=watcher, kwargs=kwargs).start()

    @property
    def finished_cb(self):
        """Callback function called when the data collection is finished"""
        return self._finished_cb

    @finished_cb.setter
    def finished_cb(self, func):
        self._finished_cb = func
        # Run immediately if already finished
        if self.done and self._finished_cb is not None:
            self._finished_cb()

    def _finish(self):
        self.done = True
        self._finished_cb()

class DataCollectionThread(threading.Thread):
    """
    Background thread that listens to zeromq events from the detector and gives a callback with the data every update_interval seconds.
    """
    def __init__(self, zc, duration, callback, update_interval=2):
        threading.Thread.__init__(self)
        self.zc = zc
        self.callback = callback
        self.duration = duration
        self.update_interval = update_interval

    def run(self):
        self.zc.read(0)
        self.zc.set_framemode(0) # Normal mode; Frame starts on FRAME_SOFTTRIG and completes after FRAME_TIME
        self.zc.fifo_reset()
        self.zc.set_framelen(self.duration)
        self. zc.start_frame()

        mca = np.zeros([192, 4096])
        start_time = tm.time()
        last_update = start_time
        while True:
            [address, msg] = self.zc.data_sock.recv_multipart()
            if (address == zclient.TOPIC_META):
                if tm.time() - start_time > self.duration - 10: # Only end if we are 10 seconds from the end (ugly fix)
                    break
            if (address == zclient.TOPIC_DATA):
                data = np.frombuffer(msg, dtype=np.uint32)
                for x in data[::2]:
                    addr = x >> 22 & (1 << 9) - 1
                    addr = addr - (int(addr/64) + 1)*32
                    energy = x & (1 << 12) - 1
                    mca[addr][energy] += 1

                if tm.time() - last_update > self.update_interval and self.update_interval >= 0:
                    last_update = tm.time()
                    self.callback(mca, time_elapsed=tm.time()-start_time, time_remaining=self.duration - (tm.time()-start_time))
                    
            if tm.time() - start_time > self.duration + 10: # Break if the detector doesn't send an end signal (ugly fix)
                break

        self.callback(mca, done=True, time_elapsed=tm.time()-start_time, time_remaining=0)

class GSD192:
    """
    Detector interface for bluesky. This should work exactly like an ophyd detector.

    Example:
    ```python
    from bluesky import RunEngine
    from bluesky.plans import count
    from gsd192_tools.bluesky import GSD192

    RE = RunEngine({})
    
    det = GSD192(ip_addr="tcp://10.0.143.160") # connect to detector at that IP
    det.configure(time=60) # Set data collection time to one minute

    RE(count([det])) # Collect data
    ```
    """
    hints = {'fields': ['total_counts']}
    """Hints for Bluesky which fields it should show in the table during data collection"""

    def __init__(self, ip_addr="tcp://10.0.143.160", name="GSD192", keep_configuration=False):
        """
        Create a GSD192 detector object that can be used with Bluesky.

        :param ip_addr: The IP Address of the detector
        :param name: The name of the detector for Bluesky purposes
        :param keep_configuration: Don't apply default configuration to detector immediately. This will result in incorrect configuration information being stored for runs.
        """
        self.name = name
        self.parent = None
        self.ip_addr = ip_addr

        self.update_interval = 2 # Update progress bar every n seconds

        self._MCA_SHAPE = [192, 4096]

        self._mca = np.zeros(self._MCA_SHAPE)
        self._strip_counts = np.zeros([192])
        self._total_counts = 0
        self._data_time = tm.time()
        self._status = Status()
        self._config = OrderedDict([
            ('shaping', {'value': 2, 'timestamp': tm.time()}),
            ('gain', {'value': 0, 'timestamp': tm.time()}),
            ('threshold', {'value': 215, 'timestamp': tm.time()}),
            ('time', {'value': 60, 'timestamp': tm.time()})
        ])

        self._CONFIG_VALUE_TEXT = {
            "shaping": ["0.25 usec", "1 usec", "0.5 usec", "2 usec"],
            "gain": ["200 keV", "100 keV", "50 keV", "25 keV"]
        }

        self._subscribers = []

        self.zc = zclient(ip_addr)
        # Apply default configuration to detector
        if not keep_configuration:
            configure(zc=self.zc)

    def _collection_callback(self, mca, done=False, time_remaining=None, time_elapsed=None):
        if done:
            self._status._finish()

        if time_remaining is not None and time_elapsed is not None:
            self._status._notify(
                fraction=time_elapsed/(time_remaining+time_elapsed),
                #time_elapsed=time_elapsed,
                #time_remaining=time_remaining,
                current=time_elapsed,
                target=time_elapsed+time_remaining,
                initial=0,
                unit="s",
                name=self.name
            )

        self._mca = mca.copy()
        self._strip_counts = np.asarray([sum(s) for s in self._mca])
        self._total_counts = sum(self._strip_counts)
        
        for subscriber in self._subscribers:
            threading.Thread(target=subscriber).start()

    def trigger(self) -> Status:
        """
        Run data collection.
        The returned status object can be used to monitor progress and register a completion callback

        :return: Status object
        """
        self._status = Status()
        self._mca = np.zeros(self._MCA_SHAPE)
        self._strip_counts = np.zeros([192])
        DataCollectionThread(self.zc, self._config['time']['value'], self._collection_callback, update_interval=self.update_interval).start()
        return self._status

    def subscribe(self, func):
        """
        Add callback function that is run every time new data is available.

        :param func: The callback function
        """
        if not func in self._subscribers:
            self._subscribers.append(func)

    def clear_sub(self, func):
        """
        Remove callback function.

        :param func: The callback function to remove
        """
        self._subscribers.remove(func)

    def read(self) -> OrderedDict:
        """
        Get the data collected in the current run.

        :return: The data as a dict
        """
        return OrderedDict([
            ('mca', {'value': self._mca.copy(), 'timestamp': self._data_time}),
            ('strip_counts', {'value': self._strip_counts.copy(), 'timestamp': self._data_time}),
            ('total_counts', {'value': self._total_counts.copy(), 'timestamp': self._data_time})
        ])

    def describe(self) -> OrderedDict:
        """
        Get dict containing info about the data fields.

        :return: The info dict
        """
        return OrderedDict([
            ('mca', {'source': "ZeroMQ server data channel", 'dtype': 'array', 'shape': self._MCA_SHAPE}),
            ('strip_counts', {'source': "ZeroMQ server data channel", 'dtype': 'array', 'shape': [192]}),
            ('total_counts', {'source': "ZeroMQ server data channel", 'dtype': 'number', 'shape': []})
        ])

    def read_configuration(self) -> OrderedDict:
        """
        Read the current configuration.

        :return: The configuration dict
        """
        return self._config.copy()

    def describe_configuration(self) -> OrderedDict:
        return OrderedDict([
            ('shaping', {'source': "No source", 'dtype': 'number', 'shape': []}),
            ('threshold', {'source': "No source", 'dtype': 'number', 'shape': []}),
            ('gain', {'source': "No source", 'dtype': 'number', 'shape': []}),
            ('time', {'source': "No source", 'dtype': 'number', 'shape': []})
        ])

    def configure(self, gain:int=None, shaping:int=None, threshold:int=None, time:int=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Configure the detector.

        :param gain: Detector gain. (0:200, 1:100, 2:50, 3:25 keV) (Default: 0)
        :param shaping: Shaping time. (0:0.25, 1:1, 2:0.5, 3:2 usec) (Default: 2)
        :param threshold: Threshold. 0-1023 (Default: 215)
        :param time: Data collection time in seconds. (Default: 60)
        """
        old = self.read_configuration()
        if gain is not None:
            if not 0 <= gain < 4:
                raise ValueError("Gain must be value 0-3")
            self._config["gain"]["value"] = gain
            self._config["gain"]["timestamp"] = tm.time()
        if shaping is not None:
            if not 0 <= shaping < 4:
                raise ValueError("Shaping must be value 0-3")
            self._config["shaping"]["value"] = shaping
            self._config["shaping"]["timestamp"] = tm.time()
        if threshold is not None:
            if not 0 <= threshold < 1024:
                raise ValueError("Threshold must be value 0-1023")
            self._config["threshold"]["value"] = threshold
            self._config["threshold"]["timestamp"] = tm.time()
        if time is not None:
            if time <= 0:
                raise ValueError("Exposure Time must be value >0")
            self._config["time"]["value"] = time
            self._config["time"]["timestamp"] = tm.time()

        if not (gain is None or shaping is None or threshold is None):
            configure(gain=gain, shaping=shaping, threshold=threshold, zc=self.zc)

        return old, self.read_configuration()

    def __repr__(self):
        return "GSD192(ip_addr={}, name={})".format(self.ip_addr, self.name)

    def __str__(self):
        string =  "{} Detector\n"
        string += "{}=========\n"
        for name, value in self._config.items():
            string += "{:<10} {}\n".format(name+":", str(value['value']) + (" ({})".format(self._CONFIG_VALUE_TEXT[name][value['value']]) if name in self._CONFIG_VALUE_TEXT.keys() else ""))

        return string.format(self.name, "="*len(self.name))[:-1]
