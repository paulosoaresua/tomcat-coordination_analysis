import inspect
import time


class ScopeTiming:

    def __init__(self):
        self.function_name = inspect.stack()[1].function
        self.start = time.time()

    def __del__(self):
        seconds = time.time() - self.start
        print(f"[{self.function_name}] Execution Time = {seconds} seconds.")
