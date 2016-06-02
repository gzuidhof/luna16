import logging

def initialize_logger(filename):
    logging.basicConfig(filename=filename,level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log(message):
    logging.info(message)
