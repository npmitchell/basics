import time

'''Tiny Timer module for timing code'''


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def get_time_ms(self):
        end = time.time()
        elapsed = (end - self.start) * 1000
        time_str = "%06f ms" % elapsed
        return time_str


if __name__ == '__main__':
    # Test the timer
    import numpy as np
    timer = Timer()
    a = 1
    for i in np.arange(1e6):
        a += 1

    print('elapsed time: ' + timer.get_time_ms())
