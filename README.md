# Audio-OFDM


## Description

This software consists of a transmitter and receiver to transmit text messages over audible spectrum (~1-3 kHz) using orthogonal frequency division multiplexing (OFDM).
OFDM is used in many wireless applications, including 5G NR, 4G LTE, and Wi-Fi.
This project is based on DrSDR's Audio-OFDM (DrSDR/Audio-OFDM) but has been heavily modified.

This code transmits and receives five OFDM symbols, the first being a pilot symbol for initial phase reference, and the last four being data symbols.
Each symbol consists of 512 BPSK modulated subcarriers, for 512 bits.
Subcarrier spacing is 3.90625 Hz and symbol duration is 0.256 s.

The pilot symbol subcarrier phases are pseudorandomly generated to reduce PAPR (peak-to-average power ratio).
The phase of each data symbol subcarrier is referenced to the respective subcarrier's phase in the previous symbol.
A 180 degree phase shift represents a binary 0, whereas no phase shift represents a binary 1.

Each symbol is repeated twice in the time domain, functioning as an extreme case of cyclic prefixing.
This repetition reduces the effects of multipath interference (echos), slightly incorrect receiver synchronization, and distortions at the symbol change times caused by the bandpass filter.

A chirp pulse before the pilot symbol is used for synchronization, so the receiver knows exactly when to begin decoding.
The reason for using the chirp signal is described by this video https://www.youtube.com/watch?v=Jyno-Ba_lKs

A steep (512th order) bandpass filter is used to greatly reduce out of band emissions.
The passband of this filter is 975 - 3025 Hz.


## Files

NOTE: This algorithm does not require the input file to be an exact output file from ofdmTX_audio.m.  A microphone recording, even with some distortions, should be sufficient.

* **ofdmTX_audio.m** - Reads in a text file (up to 256 characters) and generates a WAV file with the chirp pulse and OFDM symbols.
* **ofdmRX_Audio.m** - Reads in a WAV file and decodes the OFDM symbols back to text.  Displays the decoded text in a window and the command line.
* **ofdmtest256Char8khzAudio.wav** - sample WAV file of transmitter output/receiver input
* **textFile.txt** - sample text file for transmitter input
