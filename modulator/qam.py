import os, sys
sys.path.append(os.getcwd().split("/")[:-1])
from utils import level2bits, dB2Linear, binary2gray

from .core import Modulator
from typing import *
import math
import numpy as np


class QAM(Modulator):
    def __init__(self, M: int) -> None:
        super().__init__()

        # Check if M is an even power of 2
        power = math.log(M, 2)
        if (power % 2 != 0) or (2**power != M):
            raise ValueError("M must be a square power of 2")

        symbols = self._createConstellation(M)

        L = int(round(math.sqrt(M)))
        grayMappingIndexes = self._calculateGrayMappingIndexQAM(L)
        # noinspection PyUnresolvedReferences
        symbols = symbols[grayMappingIndexes]

        # Set the constellation
        self.setConstellation(symbols)

    @staticmethod
    def _createConstellation(M: int) -> np.ndarray:
        # Size of the square. The square root is exact
        symbols = np.empty(M, dtype=complex)
        L = int(round(math.sqrt(M)))
        for jj in range(0, L):
            for ii in range(0, L):
                symbol = complex(-(L - 1) + jj * 2, (L - 1) - ii * 2)
                symbols[ii * L + jj] = symbol

        average_energy = (M - 1) * 2.0 / 3.0 # remove /3.0 to use non-normalized energy
        # Normalize the constellation, so that the mean symbol energy is
        # equal to one.
        return symbols / math.sqrt(average_energy)

    @staticmethod
    def _calculateGrayMappingIndexQAM(L: int) -> np.ndarray:
        # Row vector with the column variation (second half of the index in
        # binary form)
        column = binary2gray(np.arange(0, L, dtype=int))

        # Column vector with the row variation
        #
        # Column vector with the row variation (first half of the index in
        # binary form)
        row = column.reshape(L, 1)
        columns = np.tile(column, (L, 1))
        rows = np.tile(row, (1, L))
        # Shift the first part by half the number of bits and sum with the
        # second part to form each element in the index matrix
        index_matrix = (rows << (level2bits(L**2) // 2)) + columns

        # Return the indexes as a vector (row order, which is the default
        # in numpy)
        return np.reshape(index_matrix, L**2)