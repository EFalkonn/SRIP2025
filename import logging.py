import logging

def lg(msg):

# Configure logging to write to a file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'  # 'a' for append, 'w' for overwrite
)



print(track)
lg(track)

# Example log messages
logging.debug('This is a debug message')


logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

Class logging:
  DEBUG = 0
 INFO =1
 WARNING = 2
ERROR = 3
CRITICAL = 4

def debug(mesg):
    logging.type=DEBUG
    write(mesg)

def info(mesg):
    logging.type=INFO
    write(mesg)

def write(mesg):
    if logging.type >= logging.level :
            print(mesg)
