import logging


class Logger:
    """
    Custom logger class that can be used to get the same logger across the application.
    """

    _logger = {}

    @classmethod
    def get_logger(cls, name='root', level=logging.DEBUG, file_name=None, **kwargs):
        if not name in cls._logger:
            # create logger
            cls._logger[name] = logging.getLogger(name)
            cls._logger[name].setLevel(level)

            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(level)

            # create formatter
            formatter = logging.Formatter('[%(asctime)s] - %(filename)s:%(funcName)s - %(levelname)s - %(message)s')

            # add formatter to ch
            ch.setFormatter(formatter)

            # add ch to logger
            cls._logger[name].addHandler(ch)

            if file_name:
                # create file handler and set level to debug
                fh = logging.FileHandler(file_name)
                fh.setLevel(level)
                fh.setFormatter(formatter)
                cls._logger[name].addHandler(fh)

        return cls._logger[name]
