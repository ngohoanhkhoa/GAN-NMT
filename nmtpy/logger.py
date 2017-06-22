# -*- coding: utf-8 -*-
import logging

def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class Logger(object):
    """Logs to stdout and to file simultaneously."""
    def __init__(self):
        pass

    def setup(self, log_file=None, timestamp=True):
        _format = '%(message)s'
        if timestamp:
            _format = '%(asctime)s ' + _format

        self.formatter = logging.Formatter(_format)
        self._logger = logging.getLogger('nmtpy')
        self._logger.setLevel(logging.DEBUG)
        self._ch = logging.StreamHandler()
        self._ch.setFormatter(self.formatter)
        self._logger.addHandler(self._ch)

        if log_file:
            self._fh = logging.FileHandler(log_file, mode='w')
            self._fh.setFormatter(self.formatter)
            self._logger.addHandler(self._fh)

    def get(self):
        return self._logger
