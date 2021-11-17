import logging
import sys

DEBUG = logging.DEBUG   # 10
DEBUG_WITH_IMAGES = 9
INFO = logging.INFO
WARNING = logging.WARNING

def get_logger(logger_name):
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)
    log = logging.getLogger(name=logger_name)
    return log
