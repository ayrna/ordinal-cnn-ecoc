import time
from typing import Optional

class Timer:
    def __init__(self, format_str: Optional[str]=None) -> None:
        self.format_str = format_str

    def __enter__(self):
        self.start_time = time.perf_counter()
    
    def __exit__(self, exc_type, *_):
        if exc_type is not None:
            return False
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

        if self.format_str is not None:
            print(self.format_str.format(self.elapsed_time))