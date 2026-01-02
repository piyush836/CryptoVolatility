import sys
import logging
from typing import Any

logger=logging.getLogger(__name__)

class CryptoException(Exception):
    def __init__(self,error_message:str,error_detail:Any=None):
        self.error_message=error_message
        self.error_detail=error_detail
        _, _, exec_tb = sys.exc_info()
        if exec_tb is not None:
            self.lineno = exec_tb.tb_lineno
            self.file_name = exec_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = None
    
    def __str__(self):
        return(   "Error occurred in python script name [{0}] line number [{1}] error message [{2}]"
        ).format(self.file_name, self.lineno, str(self.error_message))
    
if __name__ == '__main__':
    try:
        # use the configured logger (import triggers logging configuration)
        logger.info("Enter the try block")
        a = 1 / 0
        print("This will not be printed", a)
    except Exception as e:
        # Log the original exception and raise our wrapped exception with sys
        logger.exception("An error occurred in main demo")
        raise CryptoException(str(e), sys)