from .zclient import zclient
import sys
import numpy as np
import time

def parseEvent(word, timeWord=None):
    """
    An event consists of two 32-bit words. The first one contains all the data we need for the MCA file.
    _________________________________________________________________________________________________
    |               byte_0  |               byte_1  |            byte_2     |               byte_3  |
    |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
    | 0|                  address |                         tdc |                            energy |
    _________________________________________________________________________________________________
    _________________________________________________________________________________________________
    |               byte_0  |               byte_1  |            byte_2     |               byte_3  |
    |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
    | 1| 0| 0| 0|                                                                  FPGA clock ticks |
    _________________________________________________________________________________________________

    :param word: The first word of the event (the important one)
    :param timeWord: The second word of the event (will be ignored)
    :return: address, energy
    """
    # Returns address, energy
    return word >> 22 & (1 << 9) - 1, word & (1 << 12) - 1

def measure_mca_data(zc, quiet=False):
    nbr = 0
    total_len = 0
    mca = np.zeros((192, 4096), dtype=np.uint32)
    startTime = time.time()
    try:
        while True:
            [address, msg] = zc.data_sock.recv_multipart()

            if msg == b'END':
                if not quiet:
                    print("Received %s messages" % str(nbr))
                    print("Message END received")
                break
            if (address == zclient.TOPIC_META):
                if not quiet:
                    print("\nMeta data received")
                meta_data = np.frombuffer(msg, dtype=np.uint32)
                if not quiet:
                    print("Data measurement number: {}".format(meta_data))
                #np.savetxt("%s.txt"%filename,meta_data,fmt="%x")
                break

            if (address == zclient.TOPIC_STRT):
                if not quiet:
                    print("START FRAME received")
                meta_data = np.frombuffer(msg, dtype=np.uint32)
                if not quiet:
                    print(meta_data)
            if (address == zclient.TOPIC_FNUM):
                if not quiet:
                    print("fnum received")
                meta_data = np.frombuffer(msg, dtype=np.uint32)
                if not quiet:
                    print(meta_data)


            if (address == zclient.TOPIC_DATA):
                data = np.frombuffer(msg, dtype=np.uint32)
                total_len += int(len(data)/2)
                nbr += 1
                #print(data)
                for x in data[::2]:
                    # Apparently there are 12 asics, but we only want the data from odd numbered ones.
                    addr = x >> 22 & (1 << 9) - 1
                    addr = addr - (int(addr/64) + 1)*32
                    #print(x >> 22 & (1 << 9) - 1, x & (1 << 12) - 1, addr)
                    mca[addr][x & (1 << 12) - 1] += 1
                if not quiet:
                    print("\rMsg #: {0:9,d}; # Events: {1:12,d}; Time Elapsed: {2}  ".format(nbr,total_len,time.strftime("%H:%M:%S", time.gmtime(time.time()-startTime))), end="")
    except KeyboardInterrupt:
        print("")
        print("Caught Keyboard Interrupt!")
        print("Continuing to save...")

    return mca

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: gsd192-time-mca <exposure time (sec)> <filename> [connection string]")
        sys.exit()
    exposureTime = int(sys.argv[1])
    filename = sys.argv[2]

    if len(sys.argv) == 4:
        ip_addr = sys.argv[3]
    else:
        ip_addr = "tcp://10.0.143.160"
    print("Starting ZClient...")
    zc = zclient(ip_addr)
    zc.read(0)

    zc.set_framemode(0) # Normal mode; Frame starts on FRAME_SOFTTRIG and completes after FRAME_TIME
    zc.fifo_reset()
    zc.set_framelen(exposureTime)
    zc.start_frame()
	
    data = measure_mca_data(zc)

    if not filename.endswith(".mca"):
        filename = filename+".mca"

    fd = open(filename, 'wb')
    fd.write("\t#name: {}\n".format(filename))
    fd.write("\t#type: MCA\n")
    fd.write("\t#rows: 192\n")
    fd.write("\t#columns: 4096\n")
    for strip in data:
        fd.write("  ".join(map(lambda e : str(e), strip)))
        fd.write("\n")
    fd.close()

if __name__ == "__main__":
    main()
