import os, sys
import argparse
from modulator import QAM, OFDM
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # ARGUMENT
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modulator_size", default=16, type=int)
    parser.add_argument("-s", "--symbol_cnt", default=1000, type=int)
    
    parser.add_argument("-fft_size", "--fft_size", default=1024, type=int)
    parser.add_argument("-cp_size", "--cyclic_size", default=10, type=int)
    parser.add_argument("-ofdm_cnt", "--num_ofdm_symbols", default=10, type=int)
    parser.add_argument("-sub_cnt", "--num_used_subcarriers", default=600, type=int)
    
    parser.add_argument("-nv", "--noise_var", default=1e-3, type=float)
    parser.add_argument("-bw", "--bandwidth", default=5e6, type=int)
    parser.add_argument("-fd", "--doppler_freq", default=10, type=int)
    
    args = parser.parse_args()
    
    # SETTINGS
    
    ## RUN PATH
    run_dir = os.getcwd() + "/run"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    main_result_path = run_dir + "result.csv"
    if not os.path.exists(main_result_path):
        main_dict = {
            "modulator_size" : [args.modulator_size],
            "symbol_cnt" : [args.symbol_cnt],
            "fft_size" : [args.fft_size],
            "cyclic_size" : [args.cyclic_size],
            "ofdm_cnt" : [args.num_ofdm_symbols],
            "sub_cnt" : [args.num_used_subcarriers],
            "noise_var" : [args.noise_var],
            "bandwidth" : [args.bandwidth],
            "doppler_freq" : [args.doppler_freq],
            
            "BER" : [],
            "SER" : []
        }
    
    ## MODULATOR
    modulator = QAM(args.modulator_size)
    
    ## OFDM MODULATOR
    ofdm_modulator = OFDM(
        fft_size=args.fft_size, 
        cp_size=args.cyclic_size, 
        num_used_subcarriers=args.num_used_subcarriers
    )
    
    # DATA INITIALIZATION
    input_data = np.random.randint(0, args.modulator_size, size=args.symbol_cnt)
    print(f"Input Data Shape: {input_data.shape}")
    #======================================================================================================
    
    
    
    
    # SIMULATION
    
    ## CHANNEL ENCODING
    
    ## MODULATION
    modulated_data = modulator.modulate(input_data)
    print(f"QAM Data Shape: {modulated_data.shape}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_title(f"{args.modulator_size}-QAM")
    ax.plot(modulator.symbols.real, modulator.symbols.imag, "*r", label=f"{args.modulator_size}-QAM")
    ax.axis("equal")
    
    plt.savefig(run_dir + f"/{args.modulator_size}-QAM.pdf", dpi=300, format="pdf")
    plt.savefig(run_dir + f"/{args.modulator_size}-QAM.png", dpi=300, format="png")
    
    plt.close()
    
    ## OFDM MODULATION
    ofdm_modulated_data = ofdm_modulator.modulate(modulated_data)
    print(f"OFDM Data Shape: {ofdm_modulated_data.shape}")
    
    #======================================================================================================