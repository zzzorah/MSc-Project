import logging

logging.basicConfig()
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger_epoch_result = logging.getLogger('logger_epoch_result')
logger_epoch_result.setLevel(logging.DEBUG)
result_handler = logging.FileHandler('logs/epoch_result.log')
result_handler.setFormatter(logging.Formatter())
logger_epoch_result.addHandler(result_handler)

logger_debug = logging.getLogger('logger_epoch_debug')
logger_debug.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('logs/debug.log')
debug_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)6s %(message)s'))
logger_debug.addHandler(debug_handler)
