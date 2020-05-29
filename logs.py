import os
import sys
import logging


logger = logging.getLogger()
level = os.environ.get("LOGLEVEL", logging.INFO)
logger.setLevel(logging.getLevelName(level))

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
