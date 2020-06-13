import logging

import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关

logfile = os.getcwd() + '/log.txt'
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.INFO)  # 用于写到file的等级开关

formatter = logging.Formatter('%(asctime)s\t%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

