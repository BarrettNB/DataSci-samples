"""
Time tracker class.

Created on Fri Jun  7 14:25:04 2019

@author: Barrett
"""

import time

class TimeTracker:
    def __init__(self):
        self.time = time.time()

    def getElapsedTime(self):
        final_time = time.time() - self.time
        if final_time < 1:
            return '--- %s ms ---' % int(final_time*1000) #milliseconds
        elif final_time < 120.:
            return '--- %s seconds ---' % round(final_time, 2)
        elif final_time < 7200.:
            return '--- %s minutes ---' % round(final_time/60., 2)
        elif final_time < 172800:
            return '--- %s hours ---' % round(final_time/3600., 2)
        else:
            return '--- %s days ---' % round(final_time/86400., 3)