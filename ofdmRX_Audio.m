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
f0 = 300;
f1 = 3000;
% f_i = f0 + (f1-f0)*t;  % instantaneous frequency of chirp
pha_i = 2*pi*f0*t + pi*(f1-f0)*t.^2;  % instantaneous phase of chirp (2pi times integral of instantaneous freq)
chirp = sin(pha_i);  % generate chirp

Nguard = 512;   % time between pulse and preamble fft data

% DEFINE VARIABLES
Nchar = 256;  % number of characters, make power of 2
Nbits = Nchar * 8;  % number of bits
Nfft_2 = Nbits + 1024;  % half of length of FFT, bits plus guard freq bins
Nfft = (2*Nfft_2)  - 1 ;  % actual length of FFT, due to conjugate symmetry


sigtime = Nguard + 2*Nfft + 2*Nfft + 2*Nguard;  % time duration, in samples, of desired signal


% UTILIZE CHIRP FOR SYNCHRONIZATION
hpw = conj(chirp(end:-1:1)); % time reverse and conjugate chirp sync pulse
xdet = conv(hpw,x);  %convolve signal with reverse-conjugated chirp

% PLOT SIGNAL CONVOLVED WITH REVERSE-CONJUGATE CHIRP
figure(1)
plot(xdet)
title('chirp detection')
xlabel('sample')
ylabel('x[n] * chirp*[n]')


[maxv, maxi] = max(abs(xdet));  % determine max value and index from detection signal


% determine start and end samples for desired portion of audio
xstart = maxi ;
xend = xstart + sigtime ;


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


% EXTRACT PILOT TIME SIGNAL
xp0 = Nguard  + round(0.8*Nfft) + 1;  % start sample for pilot
xp1 = xp0 + Nfft - 1;  % end sample for pilot
xp = x(xp0:xp1); % extract pilot from audio


% EXTRACT DATA TIME SIGNAL
xd0 =  Nguard + 2*Nfft + round(0.8*Nfft) + 1;  % start sample for data
xd1 = xd0 + Nfft - 1;  % end sample for data
xd = x(xd0:xd1);  % extract data from audio


% PLOT EXTRACTED SIGNALS
figure(3)
x_plot = x;
xp_plot = zeros(1,length(x));
xp_plot(xp0:xp1) = 1;  % window for pilot FFT
xd_plot = zeros(1,length(x));
xd_plot(xd0:xd1) = 1;  % window for data FFT
plot(x_plot);
hold on
plot(xp_plot);
plot(xd_plot);
hold off
title('Audio samples after chirp detection pulse')
xlabel('Sample')
ylabel('x[n]')
legend('received signal','portion used for pilot FFT', 'portion used for data FFT')


% COMPUTE PILOT AND DATA SIGNAL FOURIER COEFFICIENTS WITH FFT
Xp = fft(xp);
Xp = Xp( 1:floor(end/2) );  % keep only first half of Fourier coefficients

Xd = fft(xd);
Xd = Xd( 1:floor(end/2) );  % keep only first half of Fourier coefficients


% PLOT PILOT AND DATA SPECTRA
n = length(Xd);
fvec = linspace(0,fs/2,n);
figure(4)
plot(fvec,20*log10(abs(Xp)))
hold on
plot(fvec,20*log10(abs(Xd)))
hold off
title('Pilot and data spectra')
xlabel('frequency')
ylabel('dB')
legend('Pilot FFT', 'Data FFT');


% OBTAIN PHASE FOR EACH BIT
det = abs( angle(Xp ./ Xd) );  % compute absolute phase difference between pilot and data Fourier coefficients

det = det(513:end);  % discard first 512 decoded bits (corresponding to nulled subcarriers)
det = det(1:Nbits);  % keep only bits that were transmitted


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
