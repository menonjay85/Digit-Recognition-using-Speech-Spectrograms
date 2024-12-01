[x, fs] = audioread('cleanspeech.wav');
t = (0:length(x)-1) / fs;

figure;

subplot(3, 1, 1);
plot(t, x);
title('Time Domain Waveform');
xlabel('Time (s)');
ylabel('Amplitude');

N = length(x);
X_fft = fft(x);
f = (0:N-1)*(fs/N); 
magnitude = abs(X_fft);

subplot(3, 1, 2);
plot(f(1:floor(N/2)), magnitude(1:floor(N/2)));
title('FFT of the audio file');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(3, 1, 3);
window_length = 128;
overlap = 120;
nfft = 1024;

spectrogram(x, window_length, overlap, nfft, fs, 'yaxis');
title('Spectrogram of the audio file');
colorbar;

