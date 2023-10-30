import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

from utils import randn_c, level2bits, int2bits
from modulator import QAM, OFDM
from equalizer import _calcMMSEFilter


if __name__ == "__main__":
    
    # ARGUMENT
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modulator_size", default=64, type=int)
    # parser.add_argument("-s", "--symbol_cnt", default=1000, type=int)
    
    parser.add_argument("-fft_size", "--fft_size", default=24, type=int) # 1024
    parser.add_argument("-cp_size", "--cyclic_size", default=2, type=int)
    parser.add_argument("-ofdm_cnt", "--num_ofdm_symbols", default=2, type=int)
    parser.add_argument("-sub_cnt", "--num_used_subcarriers", default=24, type=int) # 600
    
    parser.add_argument("-nv", "--noise_var", default=1e-4, type=float)
    parser.add_argument("-bw", "--bandwidth", default=5e6, type=int)
    parser.add_argument("-fd", "--doppler_freq", default=10, type=int)
    
    parser.add_argument("-nt", "--num_transmit", default=2, type=int)
    parser.add_argument("-nr", "--num_receive", default=2, type=int)
    parser.add_argument("-snr", "--snr", default=0, type=float) #0.0, 5.0, 10.0, 15.0, 20.0
    
    args = parser.parse_args()
    
    args.symbol_cnt = args.num_used_subcarriers * args.num_ofdm_symbols * args.num_transmit
    
    # SETTINGS
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
    
    # DATA INITIALIZATION
    input_data = np.random.randint(0, args.modulator_size, size=args.symbol_cnt)
    print(f"<Data> Input Data Shape: {input_data.shape}")
    #======================================================================================================
    
    
    
    # SIMULATION
    
    ## CHANNEL ENCODING
    encoded_data = input_data
    
    ## MODULATION
    modulator = QAM(args.modulator_size)
    modulated_data = modulator.modulate(encoded_data)
    print(f"<MODULATION> QAM Data Shape: {modulated_data.shape}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_title(f"{args.modulator_size}-QAM")
    ax.plot(modulator.symbols.real, modulator.symbols.imag, "*r", label=f"{args.modulator_size}-QAM")
    ax.axis("equal")
    
    plt.savefig(run_dir + f"/{args.modulator_size}-QAM.pdf", dpi=300, format="pdf")
    plt.savefig(run_dir + f"/{args.modulator_size}-QAM.png", dpi=300, format="png")
    
    plt.close()
    
    ### MAPPING
    num_elements = modulated_data.size
    
    mapped_data = (modulated_data.reshape((args.num_transmit, -1), order='F') / math.sqrt(args.num_transmit))
    
    print(f"<MAPPING> Mapping Data Shape: {mapped_data.shape}")
    
    ## OFDM MODULATION
    ofdm_modulator = OFDM(
        fft_size=args.fft_size, 
        cp_size=args.cyclic_size, 
        num_used_subcarriers=args.num_used_subcarriers
    )
    num_mapped_signal = mapped_data.shape[0]
    ofdm_modulated_data = np.zeros([num_mapped_signal, (args.fft_size + args.cyclic_size)*args.num_ofdm_symbols], dtype=complex)
    
    for idx in range(num_mapped_signal):
        _signal = mapped_data[idx]
        ofdm_modulated_data[idx] = ofdm_modulator.modulate(_signal)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(np.real(ofdm_modulated_data[idx]), np.imag(ofdm_modulated_data[idx]), 'r*')
        ax.axis('equal')
        
        plt.savefig(
            run_dir + f"/{args.fft_size}_{args.cyclic_size}_{args.num_used_subcarriers}-OFDM_{idx}.pdf", dpi=300, format="pdf"
        )
        plt.savefig(
            run_dir + f"/{args.fft_size}_{args.cyclic_size}_{args.num_used_subcarriers}-OFDM_{idx}.png", dpi=300, format="png"
        )
    print(f"<OFDM> Data Shape: {ofdm_modulated_data.shape}")
    #======================================================================================================
    
    ## CHANNEL
    
    ### CHANNEL INIT
    channel = randn_c(args.num_receive, args.num_transmit)
    print(f"<CHANNEL> CHANNEL SHAPE: {channel.shape}")
        
    print(f"<CHANNEL> No. Streaming: {args.num_transmit}")
    
    ### FADING
    
    
    ### NOISE
    noise_expected_shape = ofdm_modulated_data.shape
    awgn_noise = (randn_c(noise_expected_shape[0], noise_expected_shape[1]) * np.sqrt(args.noise_var))
    print(f"<CHANNEL> Noise Shape: {awgn_noise.shape}")
    
    ### RECEIVE
    received_signal = np.dot(channel, ofdm_modulated_data)
    print(f"<CHANNEL> Received Signal Shape: {received_signal.shape}")
    
    noise_received_signal = received_signal + awgn_noise
    print(f"<CHANNEL> Noise Received Signal Shape: {noise_received_signal.shape}")
    
    ## MMSE EQUALIZER
    W = _calcMMSEFilter(channel, args.noise_var)
    
    W = W * math.sqrt(args.num_transmit)
    print(f"<EQUALIZER> Coeficient Matrix Shape: {W.shape}")
    
    ## SIGNAL FILTERING
    filtered_signal = np.dot(W, noise_received_signal) 
    print(f"<EQUALIZER> Filtered Signal Shape: {filtered_signal.shape}")
    
    ## OFDM DEMODULATION
    ofdm_demodulated_data = np.zeros([num_mapped_signal, args.symbol_cnt//2], dtype=complex)
    
    for idx in range(num_mapped_signal):
        _signal = filtered_signal[idx]
        ofdm_demodulated_data[idx] = ofdm_modulator.demodulate(_signal)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(np.real(ofdm_demodulated_data[idx]), np.imag(ofdm_demodulated_data[idx]), 'r*')
        ax.axis('equal')
        
        plt.savefig(
            run_dir + f"/{args.fft_size}_{args.cyclic_size}_{args.num_used_subcarriers}-DEOFDM_{idx}.pdf", dpi=300, format="pdf"
        )
        plt.savefig(
            run_dir + f"/{args.fft_size}_{args.cyclic_size}_{args.num_used_subcarriers}-DEOFDM_{idx}.png", dpi=300, format="png"
        )
    print(f"<DEOFDM> Data Shape: {ofdm_demodulated_data.shape}")
    
    ## DEMAPPING
    demapped_data = ofdm_demodulated_data.flatten()
    print(f"<DEMAPPED> Data Shape: {demapped_data.shape}")
    
    ## DEMODULATION
    demodulated_data = modulator.demodulate(demapped_data)
    print(f"<DEMODULATATION> Data Shape: {demodulated_data.shape}")
    #======================================================================================================
    
    
    # EVALUATION
    input_bits = np.array(list(map(int2bits, input_data)))
    output_bits = np.array(list(map(int2bits, demodulated_data)))
    
    print(f"<EVAL> Input Bit Shape: {input_bits.shape}")
    print(f"<EVAL> Output Bit Shape: {output_bits.shape}")
    
    error_cnt = np.sum(input_bits != output_bits)
    ber = error_cnt/input_bits.shape[0]
    print(f"Number of Error bits: {error_cnt}")
    print(f"BER: {ber}")