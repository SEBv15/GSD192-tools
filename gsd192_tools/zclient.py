import zmq
import numpy as np

# Largely copied from maia_zclient_time.py

class zclient(object):
    ZMQ_DATA_PORT = "5556"
    ZMQ_CNTL_PORT = "5555"
    TOPIC_DATA = b"data"
    TOPIC_META = b"meta"
    TOPIC_STRT = b"strt"
    TOPIC_FNUM = b"fnum"
    def __init__(self, connect_str):
        self.context = zmq.Context()
        self.data_sock = self.context.socket(zmq.SUB)
        self.ctrl_sock = self.context.socket(zmq.REQ)

        self.data_sock.connect(connect_str + ":" + zclient.ZMQ_DATA_PORT)
        self.data_sock.setsockopt(zmq.SUBSCRIBE, zclient.TOPIC_DATA)
        self.data_sock.setsockopt(zmq.SUBSCRIBE, zclient.TOPIC_META)
        self.data_sock.setsockopt(zmq.SUBSCRIBE, zclient.TOPIC_STRT)
        self.data_sock.setsockopt(zmq.SUBSCRIBE, zclient.TOPIC_FNUM)

        self.ctrl_sock.connect(connect_str + ":" + zclient.ZMQ_CNTL_PORT)

    def __cntrl_recv(self):
        msg = self.ctrl_sock.recv()
        dat = np.frombuffer(msg, dtype=np.uint32)
        return dat

    def __cntrl_send(self, payload):
        self.ctrl_sock.send(np.array(payload, dtype=np.uint32))

    def destroy(self):
        self.context.destroy()

    def write(self, addr, value):
        self.__cntrl_send([0x1, int(addr), int(value)])
        self.__cntrl_recv()

    def read(self, addr):
        self.__cntrl_send([0x0, int(addr), 0x0])
        return int(self.__cntrl_recv()[2])

    def set_burstlen(self,value):
        self.write(0x8C,value)

    def set_bufsize(self,value):
        self.write(0x90,value)

    def set_rate(self,rate):
        n = int(400e6 / (rate*1e6) - 1)
        self.write(0x98,n)

    def set_framelen(self,value):
        # 40ns firmware - old
        #print "framelen: %d" %  (value*25000000)
        # self.write(0xD4,value*25000000)
        # 1 us firmware - new
        self.write(0xD4,value*1000000)

    def start_frame(self):
        self.write(0xD0,1)

    def get_framenum(self):
        val = self.read(0xD8)
        return val
		
    def set_framemode(self,value):
        self.write(0xDC,value) #0=normal mode, 1=infinite
		
    def fifo_reset(self):
        self.write(0x68,4)
        self.write(0x68,1)

    def trigger_data(self):
        self.__cntrl_send([0x2, 0x0, 0x0])
        self.__cntrl_recv()
