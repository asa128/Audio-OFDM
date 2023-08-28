# Audio-OFDM


## Description

This software consists of a transmitter and receiver to transmit text messages over audible spectrum (~1-3 kHz) using orthogonal frequency division multiplexing (OFDM).
OFDM is used in many wireless applications, including 5G NR, 4G LTE, and Wi-Fi.
This project is based on DrSDR's Audio-OFDM (DrSDR/Audio-OFDM) but has been heavily modified.

This code transmits and receives five OFDM symbols, the first being a pilot symbol for initial phase reference, and the last four being data symbols.
Each symbol consists of 512 BPSK modulated subcarriers, for 512 bits.
Subcarrier spacing is 3.90625 Hz and symbol duration (without cylic prefix) is 0.256 s.

The pilot symbol subcarrier phases are pseudorandomly generated to reduce PAPR (peak-to-average power ratio).
The phase of each data symbol subcarrier is referenced to the respective subcarrier's phase in the previous symbol.
A 180 degree phase shift represents a binary 0, whereas no phase shift represents a binary 1.

The samples at the rear of each symbol are copied and affixed to the front of their respective symbol.  This technique is known as cylic prefixing.
Near the top of each code file, this cp_duration can be adjusted from 0 (0% of the symbol samples copied, or 0 s per symbol) to 1 (100% of the symbol samples copied, or 0.256 s per symbol).
Cylic prefixing reduces the effects of multipath interference (echos), slightly incorrect receiver synchronization, and distortions at the symbol change times caused by the bandpass filter.

A chirp pulse before the pilot symbol is used for synchronization, so the receiver knows exactly when to begin decoding.
The reason for using the chirp signal is described by this video https://www.youtube.com/watch?v=Jyno-Ba_lKs

A steep (512th order) bandpass filter is used to greatly reduce out of band emissions.
This bandpass filter can be disabled in one line near the top of each code file for experimental purposes.
The passband of this filter is 975 - 3025 Hz.


## Files

NOTE: This algorithm does not require the input file to be an exact output file from ofdmTX_audio.m.  A microphone recording, even with some distortions, should be sufficient.

* **ofdmTX_audio.m** - Reads in a text file (up to 256 characters) and generates a WAV file with the chirp pulse and OFDM symbols.
* **ofdmRX_Audio.m** - Reads in a WAV file and decodes the OFDM symbols back to text.  Displays the decoded text in a window and the command line.  Also displays average and peak phase error in radians (worst case is 1.57, i.e. 180°).
* **ofdmtest256Char8khzAudio.wav** - sample WAV file of transmitter output/receiver input
* **textFile.txt** - sample text file for transmitter input
* **ofdmFullSystem.m** - simulates the full transmitter and receiver system in one run without needing to manually select the output audio.
* **ofdmFullSystem_manytrials.m** - simulates the full transmitter and receiver system over many runs to more accurately calulate the average and peak phase errors.

Note that average and peak phase errors will likely be lower in the FullSystem scripts compared to ofdm_RX_Audio because 16-bit quantization is not applied to the receive signal in the FullSystem scripts.
