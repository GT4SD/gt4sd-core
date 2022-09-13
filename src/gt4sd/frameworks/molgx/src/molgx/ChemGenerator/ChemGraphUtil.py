# -*- coding:utf-8 -*-
"""
ChemGraphUtil.py

Package for IBM Molecule Generation Experience

MIT License

Copyright (c) 2022 International Business Machines Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class NumRange(object):
    """Integer range.

    Attributes:
        min (int) min value of a range
        max (int) max value of a range
    """

    def __init__(self, rng):
        """Constructor of a range.

        Args:
            rng (NumRange, list, tuple, int) : a pair of min/max values in list or tuple,
                or an int value for the same min/max value
        """
        if isinstance(rng, NumRange):
            self.min = rng.min
            self.max = rng.max
        elif isinstance(rng, list):
            self.min = rng[0]
            self.max = rng[1]
        elif isinstance(rng, tuple):
            self.min = rng[0]
            self.max = rng[1]
        else:
            self.min = rng
            self.max = rng
        if not self.min <= self.max:
            logger.error('inconsistent min/max:%f %f', self.min, self.max)

    def __str__(self):
        return '[%d,%d]'%(self.min, self.max)

    def contains(self, val):
        """Check if a value is contained in a range.

        Args:
            val (int): integer

        Returns:
            bool: True if a value contained in a range. False otherwise.
        """
        return self.min <= val <= self.max

    def width(self):
        """Get width of the range

        Returns:
            float: width of the range
        """
        return self.max - self.min

    def union(self, rng):
        """Get a union of ranges

        Args:
            rng (NumRange): a num range

        Returns:
            NumRange: union of two ranges
        """
        return NumRange([min(self.min, rng.min), max(self.max, rng.max)])

    def intersection(self, rng):
        """Get an intersection of ranges

        Args:
            rng (NumRange): a num range

        Returns:
            NumRange: intersection of two range
        """
        if self.min > rng.max or self.max < rng.min:
            # no intersection
            return None
        else:
            return NumRange([max(self.min, rng.min), min(self.max, rng.max)])
