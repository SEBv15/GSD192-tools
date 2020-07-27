#!/usr/bin/env python3

from gsd192_tools.zclient import zclient
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    ip_addr = "tcp://10.0.143.160"
    zc = zclient(ip_addr)

    print("This monitor is pretty CPU intensive when a scan is running!")

    print("Starting ZClient...")

    plt.ion()

    x = np.zeros(192)

    last_update = time.time()

    def handle_close(evt):
        print("Terminating...")
        exit()

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', handle_close)
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, 'r-')

    c=0
    while True:
        [address, msg] = zc.data_sock.recv_multipart()
        if (address == zclient.TOPIC_STRT):
            print("START FRAME received\n\n")

        if address == zclient.TOPIC_META:
            print("Metadata received (end)")
            line1.set_ydata(x)
            ax.set_ylim(0, max(x))
            fig.canvas.draw()
            fig.canvas.flush_events()
            c=0
            x = np.zeros(192)

        if (address == zclient.TOPIC_DATA):
            data = np.frombuffer(msg, dtype=np.uint32)
            for p in data[::2]:
                # Apparently there are 12 asics, but we only want the data from odd numbered ones.
                addr = p >> 22 & (1 << 9) - 1
                addr = addr - (int(addr/64) + 1)*32
                #print(p >> 22 & (1 << 9) - 1, p & (1 << 12) - 1, addr)
                x[addr] += 1
            c += 1

            if time.time() - last_update > 10:
                print("\r# Events: {}".format(c), end="")
                line1.set_ydata(x)
                ax.set_ylim(0, max(x))
                fig.canvas.draw()
                fig.canvas.flush_events()
                last_update = time.time()

if __name__ == "__main__":
    main()
