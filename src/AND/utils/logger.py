import logging
import os
from pathlib import Path

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

prefix = Path(os.path.abspath(os.path.realpath(__file__))).parents[3]

try:
    logging.basicConfig(filename=prefix / "results/logs/logger.log",
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode="w")

    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

except FileNotFoundError:
    print(
        "Could not find logger file in {prefix}/results/logs/logger.log"
    )
    raise FileNotFoundError
