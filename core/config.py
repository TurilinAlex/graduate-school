import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

__eps = os.environ.get("EPS")
__repeat = os.environ.get("REPEAT")
__rows = os.environ.get("ROWS")
__test_size = os.environ.get("TEST_SIZE")

EPS = [int(num.strip()) for num in __eps.split(",")] if __eps is not None else None
REPEAT = [int(num.strip()) for num in __repeat.split(",")] if __repeat is not None else None
ROWS = int(__rows) if __rows is not None else None
TEST_SIZE = int(__test_size) if __test_size is not None else None
