import random
from typing import List

class RightIntervalCheck:

    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        result = [-1] * n

        # Create a list of tuples with (start, end, index) and sort by start time
        intervals_with_index = sorted((interval[0], interval[1], i) for i, interval in enumerate(intervals))

        # Function to find the right interval using linear search on sorted intervals
        def findRightIndex(target):
            for start, end, index in intervals_with_index:
                if start >= target:
                    return index
            return -1

        # Iterate through each interval to find its right interval
        for i in range(n):
            result[i] = findRightIndex(intervals[i][1])

        return result

# create dataset for logisitic regression

import random

class Dataset:

    def __init__(self) -> None:
        self.intervals = []
        self.result = []
        self.n = 0
        s = Solution()
        

    def create_dataset(self, length=10):

        #create random 2 number array with the first element greater than the second
        for i in range(length):
            start = random.randint(1, 9)
            end = random.randint(start + 1, 10)
            self.intervals.append([start, end])
        return self.intervals

s = Solution()
d = Dataset()
intervals = d.create_dataset()
print(intervals)
print(s.findRightInterval(intervals))

