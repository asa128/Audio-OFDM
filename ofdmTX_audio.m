%% OFDM ENCODER

clear
close all

fs = 8e3;  % sample rate of ofdm signal 

% GENERATE CHIRP SIGNAL FOR SYNCHRONIZATION
N = 8000;  %number of samples for chirp
t = (0:7999)/fs;  % time samples for chirp
f0 = 300;
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
Nfft_2 = Nbits + 1024;  % half of length of FFT, bits plus guard freq bins


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
xp = randn(1,Nfft_2);  % vector of length Nfft_2 with normally distributed random numbers
xp = xp / max(xp) * 12;  % normalize to maximum of 12
xp = exp(1i*xp);  % create Fourier coefficients for pilot vector
xp(1:512) = 0; xp(end-511:end) = 0; % zero out first 512 and last 512 coefficients


% MAKE BPSK DATA VECTOR,  1 = 1,  0 = -1
xd = txbits;  % put binary vector here N = 2048 bits
xd(xd == 0) = -1;
xd = [zeros(1,512)   xd    zeros(1,512) ];  %512 guard on each side


% RELATE DATA VECTOR TO PILOT VECTOR
xd = xp .* xd;


% PREPARE PILOT AND DATA VECTORS FOR IFFT
xp1 = conj( xp(end:-1:2)); % reverse and conjugate pilot vector
xp = [xp1 xp]; % concatenate reverse-conjugate pilot with pilot (to make IFFT purely real)

xd1 = conj(xd(end:-1:2)); % reverse and conjugate data vector
xd = [xd1 xd]; % concatenate reverse-conjugate data with data (to make IFFT purely real)


% GENERATE PILOT TIME VECTOR
xp = ifftshift(xp);  % rearrage zero-shifted Fourier coeffients to standard format for IFFT
xp = ifft(xp);  % IFFT to synthesize time-domain pilot vector
xp = xp / max(abs(xp));  % normalize to 1


% GENERATE DATA TIME VECTOR
xd = ifftshift(xd); % rearrage zero-shifted Fourier coeffients to standard format for IFFT
xd = ifft(xd);  % IFFT to synthesize time-domain data vector
xd = xd / max(abs(xd));  % normalize to 1


% CONCATENATE TIME SIGNALS
xt = [tdvec chirp guard xp xp xd xd tdvec];
% note: pilot and data signal are each repeated, representing an extreme case of cyclic prefixing
% cyclic prefixing provides stronger immunity from errors due to multipath and imprecise synchronization


% FILTER OUTPUT TO ENSURE BANDWIDTH, REDUCE OUT OF BAND SIGNALS
hlpf = fir1(64,0.9);  % FIR filter impulse response, 64th order, low-pass with cutoff of 0.9 rad/sample (3.6 kHz)
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
Xp = fft(xp);
Xp = Xp( 1:floor(end/2) );

figure(2)
plot(abs(Xp))
title('Magnitude Spectrum of Pilot')
xlabel('Fourier coefficient')

Xd = fft(xd);
Xd = Xd( 1:floor(end/2) );


det = angle(Xp ./ Xd ); % phase difference between pilot and data

det = abs(det);
n = length(det);
f = linspace(0,fs/2,n);

thres = pi/2;

% det = ( det >= thres) .* 1   +  (det < thres) .* 0;


figure(3)
plot(f,det)
title('phase results')
xlabel('frequency')
ylabel('phase')


bits = det(513:end-511);

figure(4)
plot(bits)
title('bits')
xlabel('bit number')
ylabel('bit phase')
