# This file was created with the assistance of OpenAI's ChatGPT, which helped generate parts of the code and documentation.

import logging

def get_logger(debug=True, name=None):
    # If no name is provided, use a default name or derive from __file__ in the caller
    if name is None:
        name = "default_logger"

    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger is already configured, do not add new handlers
        return logger

    # Setting up level
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optionally add a file handler
    fh = logging.FileHandler('logfile.log')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
