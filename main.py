import os, sys
import argparse
from modulator import QAM
import numpy as np

if __name__ == "__main__":
    
    # ARGUMENT
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modulator_size", default=16, type=int)
    parser.add_argument("-s", "--symbol_cnt", default=1000, type=int)
    
    args = parser.parse_args()
    
    # SETTINGS
    modulator = QAM(args.modulator_size) # Modulator
      
    
    # DATA INITIALIZATION
    input_data = np.random.randint(0, args.modulator_size, size=args.symbol_cnt)
    
    # CHANNEL ENCODING
    
    # MODULATION
    modulated_data = modulator.modulate(input_data)
    
    print(modulated_data)