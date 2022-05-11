# import bach_utils.list as utlst
# import bach_utils.sorting as utsrt

# l = ["3", "2", "1"]

# print(utlst.get_latest(l))
# print(l)
# print(utlst.get_first(l))
# print(l)
# print(utlst.get_sorted(l, utsrt.sort_nicely))
# print(l)
import logging
logger = logging.getLogger("self-play-rl")

def testing():
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")