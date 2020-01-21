import progressbar
import time

def main():
    print('test')
    bar = progressbar.progbar(10,display='command',clear_display=True)
    for i in range(10):
        time.sleep(0.1)
        bar.update(i)
    del bar

    print('test2')
    bar = progressbar.progbar(10,display='command',clear_display=False)
    for i in range(10):
        time.sleep(0.1)
        bar.update(i)
    del bar

    print('test3')
    bar = progressbar.progbar(10,display='command',clear_display=True)
    for j in range(10):
        prog = progressbar.progbar(10,display='command',clear_display=True)
        for i in range(10):
            time.sleep(0.1)
            prog.update(i)
        del prog
        bar.update(j)
    del bar
   
if __name__=="__main__":
    main()
