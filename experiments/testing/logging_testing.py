# # https://coralogix.com/blog/python-logging-best-practices-tips/
# import logging
# # logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

# # DEBUG
# # INFO
# # WARNING
# # ERROR

# def testing():
#     logging.debug('This message should go to the log file')
#     logging.info('So should this')
#     logging.warning('And this, too')
#     logging.error('And non-ASCII stuff, too, like Øresund and Malmö')


# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

# https://stackoverflow.com/questions/15727420/using-logging-in-multiple-modules

import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)




def testing(logger):
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")


if __name__ == "__main__":
    from logging_testing import *
    logger = logging.getLogger("My_app")
    logger.setLevel(logging.DEBUG)
    
    # print(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    testing(logger)