import sys, time
from IPython.display import display
from IPython.display import clear_output
class progbar:
    def __init__(self, period=100, bars=32, display='command',clear_display=True):
        assert display in ['notebook','command'], 'display type: [notebook, command]'

        self._period  = period
        self.bars     = bars
        self.active   = True
        self.start    = time.time()
        self.display  = display
        self.clear_disp = clear_display
        self.elapsed = 0
        self.initial_update = True

    def dispose(self):
        if self.active:
            self.active = False
            timeinfo = self.calc_time(self.elapsed)
            self.update(self._period,info='Total time:{0}'.format(timeinfo))
            sys.stdout.write("\n")

    def __del__(self):
        self.dispose()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.dispose()

    def period(self):
        return self._period

    def update(self, tick,info=''):
        rate = tick / self._period

        # progress rate
        str = "{0:7d}% ".format(int(rate*100))

        # progress bar
        bar_prog = int(rate * self.bars)
        str += "|"
        str += "#" * (            bar_prog)
        str += "-" * (self.bars - bar_prog)
        str += "|"

        # calc end
        self.elapsed = time.time() - self.start
        predict = (self.elapsed * (1 - rate) / rate) if not rate == 0.0 else 0
        timeinfo = self.calc_time(predict)
        str += timeinfo

        if self.display == 'notebook':
            if self.clear_disp:
                clear_output()
            display("{0} {1}".format(str,info))
        if self.display == 'command':
            if self.clear_disp and not(self.initial_update):
                sys.stdout.write("\033[1A\033[2K\033[G")
            sys.stdout.write("{0} {1}\n".format(str,info))
            sys.stdout.flush()

        self.initial_update = False
    
    def calc_time(self,t):
        s = int(t)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        timeinfo = " {day}day{hour:3d}:{minute:02d}:{sec:02d}".format(day=d,hour=h,minute=m,sec=s)
        return timeinfo
