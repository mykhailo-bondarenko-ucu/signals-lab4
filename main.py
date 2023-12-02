import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq, ifft
from scipy.signal import butter, buttord, lfilter, freqz, cheby1, cheb1ord


def H_omega(b, a, omega):
    """ H(e^{j * omega}) """
    num = b @ np.exp(-1j * np.arange(len(b)) * omega)
    denom = a @ np.exp(-1j * np.arange(len(a)) * omega)
    return num / denom


def compute_snr(signal, noise, prefix=''):
    noise_power = np.mean(noise**2)
    signal_power = np.mean(signal**2)
    print(f"{prefix} SNR: {signal_power/noise_power}")
    print(f"{prefix} SNR(dB): {20 * np.log10(signal_power/noise_power)}")


def task_1():
    duration_s, sample_rate_hz, amp = 5, 128, 1
    pulse_start, pulse_duration = 3, 0.1
    noise_std = 0.4

    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate_hz), endpoint=False)
    fft_freqs = fftfreq(time_s.shape[0], d=1/sample_rate_hz)

    plt.figure(figsize=(12, 10))

    signal = np.zeros_like(time_s)
    start_index = int(pulse_start * sample_rate_hz)
    end_index = int((pulse_start + pulse_duration) * sample_rate_hz)
    signal[start_index:end_index] = amp

    plt.subplot(3, 2, 1)
    plt.plot(time_s, signal)
    plt.title('Time Domain: Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True)

    signal_fft = fft(signal) / time_s.shape[0]
    signal_amps = np.abs(signal_fft)

    plt.subplot(3, 2, 2)
    plt.plot(fft_freqs[:len(fft_freqs)//2], signal_amps[:len(signal_amps)//2])
    plt.title('Amplitude Spectrum: Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    noise = np.random.normal(0, noise_std, size=time_s.shape)

    plt.subplot(3, 2, 3)
    plt.plot(time_s, noise)
    plt.title('Time Domain: Noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True)

    noise_fft = fft(noise) / time_s.shape[0]
    noise_amps = np.abs(noise_fft)

    plt.subplot(3, 2, 4)
    plt.plot(fft_freqs[:len(fft_freqs)//2], noise_amps[:len(noise_amps)//2])
    plt.title('Amplitude Spectrum: Noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    signal_w_noise = signal + noise

    plt.subplot(3, 2, 5)
    plt.plot(time_s, signal_w_noise)
    plt.title('Time Domain: Signal with noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True)

    signal_w_noise_fft = fft(signal_w_noise) / time_s.shape[0]
    signal_w_noise_amps = np.abs(signal_w_noise_fft)

    plt.subplot(3, 2, 6)
    plt.plot(fft_freqs[:len(fft_freqs)//2], signal_w_noise_amps[:len(signal_w_noise_amps)//2])
    plt.title('Amplitude Spectrum: Signal with noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.savefig("task_1_signals.png")
    plt.tight_layout()
    plt.show()

    # filter parameters
    cutoff_freq_hz = 5
    stopband_freq_hz = cutoff_freq_hz * 1.28

    lowest_butterworth_order, butterworth_natural_frequency = buttord(
        wp=cutoff_freq_hz,         # cut-off freq
        ws=stopband_freq_hz,       # stopband freq
        gpass=3,
        gstop=20,
        analog=False, fs=sample_rate_hz
    )
    print(f"{lowest_butterworth_order = }")
    print(f"{butterworth_natural_frequency = }")

    # N = min(max(N, 5), 10)
    assert 5 <= lowest_butterworth_order <= 10

    b, a = butter(
        lowest_butterworth_order,
        butterworth_natural_frequency,
        btype='lowpass',
        fs=sample_rate_hz
    )

    plt.figure(figsize=(12, 10))

    filtered_signal = lfilter(b, a, signal_w_noise)

    plt.subplot(2, 2, 1)
    plt.plot(time_s, filtered_signal)
    plt.title('Time Domain: Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True)

    filtered_signal_fft = fft(filtered_signal) / time_s.shape[0]
    filtered_signal_amps = np.abs(filtered_signal_fft)

    plt.subplot(2, 2, 2)
    plt.plot(fft_freqs[:len(fft_freqs)//2], filtered_signal_amps[:len(filtered_signal_amps)//2])
    plt.title('Amplitude Spectrum: Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    omega = 2 * np.pi * cutoff_freq_hz / sample_rate_hz
    print(f"Cut-off frequecy H(w) ({cutoff_freq_hz:.2f} Hz) (dB):", 20 * np.log10(np.abs(H_omega(b, a, omega))))
    omega = 2 * np.pi * (stopband_freq_hz) / sample_rate_hz
    print(f"Stop-band frequecy H(w) ({stopband_freq_hz:.2f} Hz) (dB):", 20 * np.log10(np.abs(H_omega(b, a, omega))))

    # calculate frequency response (rads/sample)
    f_resp_freqs_rad_p_s, filter_response = freqz(b, a, worN=8000)
    # convert to Hz
    f_resp_freqs_hz = f_resp_freqs_rad_p_s * sample_rate_hz / (2 * np.pi)

    plt.subplot(2, 1, 2)
    plt.plot(f_resp_freqs_hz, np.abs(filter_response))
    plt.title('Frequency Response of the Butterworth Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)

    plt.savefig("task_1_filter.png")
    plt.tight_layout()
    plt.show()

    compute_snr(signal, noise, "Original")
    compute_snr(lfilter(b, a, signal), lfilter(b, a, noise), "Filtered")


def task_2():
    duration_s, sample_rate_hz, signal_amp, signal_freq = 1, 128, 1, 10
    noise_std = 2

    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate_hz), endpoint=False)
    fft_freqs = fftfreq(time_s.shape[0], d=1/sample_rate_hz)

    plt.figure(figsize=(12, 10))

    signal = signal_amp * np.sin(2 * np.pi * signal_freq * time_s)

    plt.subplot(3, 2, 1)
    plt.plot(time_s, signal)
    plt.title('Time Domain: Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True)

    signal_fft = fft(signal) / time_s.shape[0]
    signal_amps = np.abs(signal_fft)

    plt.subplot(3, 2, 2)
    plt.plot(fft_freqs[:len(fft_freqs)//2], signal_amps[:len(signal_amps)//2])
    plt.title('Amplitude Spectrum: Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    noise = np.random.normal(0, noise_std, size=time_s.shape)

    plt.subplot(3, 2, 3)
    plt.plot(time_s, noise)
    plt.title('Time Domain: Noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True)

    noise_fft = fft(noise) / time_s.shape[0]
    noise_amps = np.abs(noise_fft)

    plt.subplot(3, 2, 4)
    plt.plot(fft_freqs[:len(fft_freqs)//2], noise_amps[:len(noise_amps)//2])
    plt.title('Amplitude Spectrum: Noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    signal_w_noise = signal + noise

    plt.subplot(3, 2, 5)
    plt.plot(time_s, signal_w_noise)
    plt.title('Time Domain: Signal with noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True)

    signal_w_noise_fft = fft(signal_w_noise) / time_s.shape[0]
    signal_w_noise_amps = np.abs(signal_w_noise_fft)

    plt.subplot(3, 2, 6)
    plt.plot(fft_freqs[:len(fft_freqs)//2], signal_w_noise_amps[:len(signal_w_noise_amps)//2])
    plt.title('Amplitude Spectrum: Signal with noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.savefig("task_2_signals.png")
    plt.tight_layout()
    plt.show()

    for title, (wp, ws) in [
        ("lowpass", (11.1, 11.1*1.05)),
        ("highpass", (8.5*1.05, 8.5)),
        ("bandpass", ((9*1.005, 11), (9, 11*1.005))),
    ]:
        print(f"\n\n\nFilter: {title}")
        print(f"wp {wp}, ws: {ws}")
        lowest_chebyshev_order, chebyshev_natural_frequency = cheb1ord(
            wp=wp,         # cut-off freq
            ws=ws,         # stopband freq
            gpass=3,
            gstop=20,
            analog=False, fs=sample_rate_hz
        )
        print(f"{lowest_chebyshev_order = }")
        print(f"{chebyshev_natural_frequency = }\n")

        # N = min(max(N, 5), 10)
        assert 5 <= lowest_chebyshev_order <= 10

        b, a = cheby1(
            lowest_chebyshev_order,
            3,
            chebyshev_natural_frequency,
            btype=title, fs=sample_rate_hz
        )

        plt.figure(figsize=(12, 10))

        filtered_signal = lfilter(b, a, signal_w_noise)

        plt.subplot(2, 2, 1)
        plt.plot(time_s, filtered_signal)
        plt.title(f'Time Domain: Filtered Signal ({title})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        plt.grid(True)

        filtered_signal_fft = fft(filtered_signal) / time_s.shape[0]
        filtered_signal_amps = np.abs(filtered_signal_fft)

        plt.subplot(2, 2, 2)
        plt.plot(fft_freqs[:len(fft_freqs)//2], filtered_signal_amps[:len(filtered_signal_amps)//2])
        plt.title(f'Amplitude Spectrum: Filtered Signal ({title})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # calculate frequency response (rads/sample)
        f_resp_freqs_rad_p_s, filter_response = freqz(b, a, worN=8000)
        # convert to Hz
        f_resp_freqs_hz = f_resp_freqs_rad_p_s * sample_rate_hz / (2 * np.pi)

        plt.subplot(2, 1, 2)
        plt.plot(f_resp_freqs_hz, np.abs(filter_response))
        plt.title(f'Frequency Response of the chebyshev Filter ({title})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)

        plt.savefig(f"task_2_filter_{title}.png")
        plt.tight_layout()
        plt.show()

        # omega = 2 * np.pi * cutoff_freq_hz / sample_rate_hz
        # print(f"Cut-off frequecy H(w) ({cutoff_freq_hz:.2f} Hz) (dB):", 20 * np.log10(np.abs(H_omega(b, a, omega))))

        compute_snr(signal, noise, "Original")
        compute_snr(lfilter(b, a, signal), lfilter(b, a, noise), "Filtered")


def task_3():
    pass


def task_4():
    pass


def main():
    # task_1()
    task_2()
    # task_3()
    # task_4()

if __name__ == "__main__":
    main()
