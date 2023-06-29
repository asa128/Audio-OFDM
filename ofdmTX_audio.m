%% OFDM ENCODER

clear
close all

fs = 8e3;  % sample rate of ofdm signal 

% GENERATE CHIRP SIGNAL FOR SYNCHRONIZATION
N = 8000;  %number of samples for chirp
t = (0:7999)/fs;  % time samples for chirp
f0 = 1000;
f1 = 3000;
% f_i = f0 + (f1-f0)*t;  % instantaneous frequency of chirp
pha_i = 2*pi*f0*t + pi*(f1-f0)*t.^2;  % instantaneous phase of chirp (2pi times integral of instantaneous freq)
chirp = sin(pha_i);  % generate chirp


% GENERATE VECTORS OF SILENCE
guardN = 512;  % number of samples to delay after chirp
guard = zeros(1,guardN);  % delay of silence after chirp

td = 2; % number of seconds for silence at beginning and end of audio file
tdsamples = round(td * fs);
tdvec = zeros(1,tdsamples);


% DEFINE VARIABLES
Nchar = 256;  % number of characters, make power of 2
Nbits = Nchar * 8;  % number of bits
Nfft_2 = Nbits/4 + 512;  % half of length of FFT, bits plus guard freq bins


% GET MESSAGE FROM TXT FILE, CONVERT TO BITS
fid = fopen('textfile.txt');
mtxt = fread(fid);
fclose(fid);
mtxt = mtxt'; % convert column vector of text to row vector

% force message to 256 characters by adding or deleting characters
if length(mtxt) >= Nchar
    mtxt = mtxt(1:Nchar);
else
    z = length(mtxt);
    z = Nchar - z;
    mtxt = [mtxt zeros(1,z)];
end

mbits = dec2bin(mtxt,8); % convert each message character to binary
mbits = reshape(mbits',1,[]); % rearrange mbits char array to single element

% create vector of bits to transmit based on mbits char element
txbits = zeros(1,Nbits);
for k = 1:Nbits
    if mbits(k) == '1'
        txbits(k) = 1;
    else
        txbits(k) = 0;
    end
end


% MAKE PILOT VECTOR OF RANDOM PHASE ROTATIONS FOR THE FOURIER COEFFICIENTS
% random phase rotation reduces PAPR (peak-to-average power ratio) compared to all pilots having the same phase
Xp = randn(1,Nfft_2);  % vector of length Nfft_2 with normally distributed random numbers
Xp = Xp / max(Xp) * 12;  % normalize to maximum of 12
Xp = exp(1i*Xp);  % create Fourier coefficients for pilot vector
Xp(1:256) = 0; Xp(end-255:end) = 0; % zero out first 256 and last 256 coefficients


% MAKE BPSK DATA VECTOR,  1 = 1,  0 = -1
Xd = txbits;  % put binary vector here N = 2048 bits
Xd(Xd == 0) = -1;


% SPLIT DATA UP INTO 4 SYMBOLS
SYMS = zeros(5,1024);  % array for symbol Fourier coefficients, 5 symbols (rows), 1024 coefficients (columns)

SYMS(1,:) = Xp;  % make first symbol the pilot

for k = [2 3 4 5]  % loop over each data symbol
    SYMS( k, 257:(end-256) ) = Xd( (512*(k-2)+1) : (512*(k-1)) );  % set data symbol Fourier coefficients, with 256 null coefficients on each side
    SYMS(k,:) = SYMS(k-1,:) .* SYMS(k,:);  % relate phase to previous symbol
end


% PREPARE PILOT AND DATA VECTORS FOR IFFT
SYMS_RC = zeros(5,1023);  % array for reverse-conjugated symbol Fourier coefficients

for k = [1 2 3 4 5]  % loop over each symbol
    SYMS_RC(k,:) = conj( SYMS(k,end:-1:2) ); % reverse and conjugate symbol Fourier coefficicents
end

SYMS = [SYMS SYMS_RC];   % concatenate reverse-conjugate symbols with symbols (to make IFFTs purely real)


% GENERATE SYMBOL TIME VECTORS
syms = zeros(5, 2047);  % array to store symbol time vectors

for k = [1 2 3 4 5]  % loop over each symbol
    syms(k,:) = ifft( SYMS(k,:) );  % IFFT to synthesize time-domain symbol
    syms(k,:) = syms(k,:) / max( abs(syms(k,:) ) );  % normalize to 1
end


% CONCATENATE TIME SIGNALS
syms = [syms syms];  % repeat each symbol in the time domain, representing an extreme case of cyclic prefixing
% cyclic prefixing provides stronger immunity from errors due to multipath and imprecise synchronization

symvec = reshape(syms',1,[]);  % reshape time-domain symbols into a single row vector

xt = [tdvec chirp guard symvec tdvec];  % put everything together


% FILTER OUTPUT TO ENSURE BANDWIDTH, REDUCE OUT OF BAND SIGNALS
fc_lo = 975;  % low cutoff frequency in Hz
fc_hi = 3025;  % high cutoff frequency in Hz

rsc_lo = fc_lo * 2 / 8000;  % convert low cutoff frequency to rads/sample/pi
rsc_hi = fc_hi * 2 / 8000;  % convert high cutoff frequency to rads/sample/pi

hlpf = fir1(512,[rsc_lo rsc_hi]);  % FIR filter impulse response, 512th order, bandpass
xt = conv(hlpf,xt);  % convolve signal with FIR filter impulse response
xt = xt / max(abs(xt));  % normalize to 1


% WRITE TO FILE
datafile = real(xt).';  % convert time signal from row vector to column vector for audiowrite
audiowrite('ofdmtest256Char8khzAudio.wav',datafile,fs,'BitsPerSample',16);  % write audio file


% PLOT TRANSMITTED SIGNAL
figure(1)
plot(real(xt))
hold on
plot(imag(xt))
hold off
title('Transmitted Signal')
xlabel('Sample')
ylabel('x[n]')
legend('real','imag')


% PLOT MAGNITUDE SPECTRUM OF PILOT
figure(2)
plot(abs(Xp))
title('Magnitude Spectrum of Pilot')
xlabel('Fourier coefficient')


% CALCULATE PEAK-TO-AVERAGE POWER RATIO (PAPR) OVER OFDM SYMBOLS
pwr_i = xt(24512:44992).^2;  % array of instantaneous power during OFDM symbols

PAPR = max(pwr_i)/mean(pwr_i)  % PAPR in absolute units (W/W)
PAPR_db = 10*log10(PAPR)  % PAPR in dB