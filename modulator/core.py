import os, sys
sys.path.append("/".join(os.getcwd().split("/")[:-1]))

from typing import *
from utils import level2bits

import numpy as np
import matplotlib.pyplot as plt


class Modulator:
    def __init__(self) -> None:
        self.M: int = 0 
        self.K: int = 0
        self.symbols: np.ndarray = np.array([])
        self.name = None

    def setConstellation(self, symbols: np.ndarray) -> None:
        M = symbols.size
        self._M = M
        self._K = np.log2(M)
        self.symbols = symbols

    def plotConstellation(self, path = None) -> None:  # pragma: no cover
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.symbols.real, self.symbols.imag)
        ax.axis('equal')
        ax.grid()

        formatString = "{0:0=" + str(level2bits(self._M)) + "b} ({0})"

        index = 0
        for symbol in self.symbols:
            ax.text(
                symbol.real,  # Coordinate X
                symbol.imag + 0.03,  # Coordinate Y
                formatString.format(index, format_spec="0"),  # Text
                verticalalignment='bottom',  # From now on, text properties
                horizontalalignment='center')
            index += 1

        if not path:
            plt.show()
        else:
            plt.savefig(path)

    def modulate(self, inputData: Union[int, np.ndarray]) -> np.ndarray:
        try:
            return self.symbols[inputData]
        except IndexError:
            raise ValueError("Input data must be between 0 and 2^M")

    def demodulate(self, receivedData: np.ndarray) -> np.ndarray:
        shape = receivedData.shape
        reshaped_received_data = receivedData.flatten()

        constellation = np.reshape(self.symbols, [self.symbols.size, 1])
        output = np.abs(constellation - reshaped_received_data).argmin(axis=0)
        output.shape = shape

        return output