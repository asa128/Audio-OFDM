%% OFDM DECODER

clear
close all

% PROMPT USER TO SELECT A WAV FILE
FILTERSPEC = '.wav';
TITLE = 'Pick a Recorded OFDM IQ wav file';
FILE = 'ofdmtest256Char8khzAudio';
[FILENAME, PATHNAME, FILTERINDEX] = uigetfile(FILTERSPEC, TITLE, FILE);
xstring = [PATHNAME FILENAME];
[x,fswav] = audioread(xstring);  % read in file and its sample rate


x = x(:,1);  % if audio is multichannel, discard all but first channel
x = x.';  % convert column vector of audio samples to row vector


% RESAMPLE VECTOR OF AUDIO SAMPLES, IF NECESSARY
fs = 8e3;  % sample rate of ofdm signal
if fswav ~= fs
    n = gcd(fs,fswav);
    p = fs / n;
    q = fswav / n;
    x = resample(x,p,q);   % resample file to expected sample rate
end


% GENERATE CHIRP SIGNAL FOR SYNCHRONIZATION
N = 8000;  % number of samples for chirp
t = (0:7999)/fs;  % time samples for chirp
f0 = 1000;
f1 = 3000;
% f_i = f0 + (f1-f0)*t;  % instantaneous frequency of chirp
pha_i = 2*pi*f0*t + pi*(f1-f0)*t.^2;  % instantaneous phase of chirp (2pi times integral of instantaneous freq)
chirp = sin(pha_i);  % generate chirp

Nguard = 512;   % time between pulse and preamble fft data


% DEFINE VARIABLES
Nchar = 256;  % number of characters, make power of 2
Nbits = Nchar * 8;  % number of bits
Nfft_2 = Nbits/4 + 512;  % half of length of FFT, bits plus guard freq bins
Nfft = (2*Nfft_2)  - 1 ;  % actual length of FFT, due to conjugate symmetry


% UTILIZE CHIRP FOR SYNCHRONIZATION
hpw = conj(chirp(end:-1:1)); % time reverse and conjugate chirp sync pulse
xdet = conv(hpw,x);  %convolve signal with reverse-conjugated chirp
[maxv, maxi] = max(abs(xdet));  % determine max value and index from detection signal

% determine start and end samples for desired portion of audio
xstart = maxi ;
sigtime = Nguard + 2*Nfft + 4*2*Nfft + 2*Nguard;  % time duration, in samples, of desired signal
xend = xstart + sigtime ;


% PLOT SIGNAL CONVOLVED WITH REVERSE-CONJUGATE CHIRP
figure(1)
plot(xdet)
title('chirp detection')
xlabel('sample')
ylabel('x[n] * chirp*[n]')


% PLOT SIGNAL AND DETECTED CHIRP END LOCATION
figure(2)
yplot = zeros(1,length(xdet));
yplot(maxi) = 2;
plot(x)
hold on
plot(yplot)
hold off
title('Detected end location of chirp')
xlabel('sample')
ylabel('x[n]')
legend('signal','detected chirp end location')


x = x(xstart:xend);  % obtain desired portion of audio


% EXTRACT TIME SIGNALS FOR EACH SYMBOL
sym_start = zeros(1,5);  % symbol start times
sym_end = zeros(1,5);  % symbol end times
syms = zeros(5,2047);  % array for time domain symbols, each symbol is a row

for k = [1 2 3 4 5]  % loop over each symbol
    sym_start(k) = Nguard  + 2*(k-1)*Nfft + round(0.8*Nfft) + 1;
    sym_end(k) = sym_start(k) + Nfft - 1;
    syms(k,:) = x(sym_start(k):sym_end(k));
end


% PLOT EXTRACTED SIGNALS
figure(3)
fft_win = zeros(1,length(x));
for k = [1 2 3 4 5]
    fft_win(sym_start(k):sym_end(k)) = 1;  % windows for FFTs
end

plot(x);
hold on
plot(fft_win);
hold off
title('Audio samples after chirp detection pulse')
xlabel('Sample')
ylabel('x[n]')
legend('received signal', 'windows used for FFTs')


% COMPUTE PILOT AND DATA SIGNAL FOURIER COEFFICIENTS WITH FFT
SYMS = zeros(5,2047);

for k = [1 2 3 4 5]  % loop over each symbol
    SYMS(k,:) = fft( syms(k,:) );  % FFT
end


SYMS = SYMS(:,1:floor(end/2));  % keep only first half of Fourier coefficients


% PLOT PILOT AND DATA SPECTRA
n = length( SYMS(1,:) );
fvec = linspace(0,fs/2,n);
figure(4)
plot(fvec,20*log10(abs( SYMS(1,:) )))
hold on
plot(fvec,20*log10(abs( SYMS(2,:) )))
hold off
title('Pilot and data spectra')
xlabel('frequency')
ylabel('dB')
legend('Pilot FFT', 'Data FFT');


% OBTAIN PHASE FOR EACH BIT
det = zeros(4,1023);

% compute abosolute phase difference between successive symbols
for k = [2 3 4 5]  % loop over each data symbol
    det(k-1,:) = abs( angle( SYMS(k-1,:) ./ SYMS(k,:) ) );  % calculate absolute phase difference from previous symbol
end

det = det(:,257:end);  % discard first 256 bits of each data symbol (corresponding to nulled subcarriers)
det = det(:,1:Nbits/4);  % keep only bits that were transmitted (512 bits per data symbol)

det = reshape(det',1,[]);  % reshape into single row vector


% PLOT DECODED BIT PHASES
figure(5)
plot(det)
title('demodulate phase vector')
xlabel('bit number')
ylabel('bit phase')


% DETERMINE BIT VALUES BASED ON THRESHOLD
thres = pi/2;  % threshold angle for bit determination

det(det <= thres) = 1;  % 1 bits have no phase inversion (i.e., angle less than threshold)
det(det > thres) = 0;  % 0 bits have phase inversion (i.e., angle greater than threshold)


% PLOT DECODED BITS
figure(6)
plot(det)
title('decoded bits vector')
xlabel('bit number')
ylabel('bit value')


% CONVERT BINARY BACK TO CHARACTERS
det = reshape(det,8,[]);  % reshape decoded bits into array with columns of 8
det = det';  % transpose decoded bit array into rows of 8

charmes = zeros(1,Nchar);  % array to store characters

w = 2.^(7:-1:0);  % powers of 2 to convert binary to decimal

for k = 1:Nchar  % loop over each element of character array
    x = det(k,:);  % extract corresponding row in bit array
    x = sum(x .* w);  % compute decimal value of the character
    charmes(k) = x;  % store to kth element of character array
end


% DISPLAY RESULTS
clc  % clear command window
xstr = char(charmes)  % display decoded message in command window
msgbox(xstr,'replace')  % display decoded message in message box
