import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from music21 import *
from scipy.signal import savgol_filter


'''
returns sorted list of (idx, value)
'''
def peaks(seq):
    data = []
    for i, x in enumerate(seq):
        if i <= 1 or i >= len(seq) - 2:
            continue
    #    if seq[i - 2] < seq[i - 1] < x and seq[i + 2] < seq[i + 1] < x:
        data.append((i, x))
    return sorted(data, key=lambda x: -x[1])


def db_to_vol(db,freq):
    transformed = np.fix(float((127 - 3 * abs(db))) * np.exp(-freq / 2000))
    return transformed if transformed > 32 else 0


def keydiff(freq1, freq2): 
    return abs(12 * np.log2(freq1 / freq2))


def make_note(freq, dur, vol):
    n = note.Note()
    p = pitch.Pitch()
    p.frequency = freq
    n.pitch = p
    n.duration = duration.Duration(dur)
    n.volume.velocity = vol
    return n


def make_stream(top_freqs, keydiff_threshold=1):
    s = stream.Stream()
    freqs = np.array([f for (f, i) in top_freqs])
    intensities = np.array([i for (f, i) in top_freqs])
    default_dur = 0.005

    for voice, ints in zip(freqs.T, intensities.T):
        
        par = stream.Part()
        last_freq = voice[0]
        last_tense = ints[0]
        dur = default_dur
        vol = db_to_vol(ints[0],last_freq)

        for note_idx in range(1, len(voice)):
            if keydiff(voice[note_idx], last_freq) >= keydiff_threshold or ints[note_idx]>last_tense:
                
                n = make_note(last_freq, dur, vol)
                par.append(n)

                # reset
                last_freq = voice[note_idx]
                last_tense= ints[note_idx]
                dur = default_dur
                vol = db_to_vol(ints[note_idx], last_freq)
            else:
                dur += default_dur

        n = make_note(last_freq, dur, vol) 
        par.append(n)
        s.insert(0, par) 
    return s  


ZERO_VOLUME = -80  # dB

def mute_low_volume(seq):
    return [x if x > -60 else ZERO_VOLUME for x in seq]


def make_bin2freq(sr, n_fft):
    return dict(enumerate(librosa.fft_frequencies(sr=sr, n_fft=n_fft)))


'''
return [(pitches, intensities), ... for each time step]
'''
def compute_top_frequencies(spec, n_peaks):
    bin2freq = make_bin2freq(sr=44100, n_fft=4096)
    top_freqs = []
    for time_slice in spec.T:
        pitches = []
        intensities = []

        # remove some high frequencies
        
        time_slice = time_slice[:2000]

        # silence murmurs
        time_slice = mute_low_volume(time_slice)

        # low cut
        for i in range(6):
            time_slice[i] = ZERO_VOLUME

        # smooth the frequencies
        time_slice = savgol_filter(time_slice, 9, 3)
    
        # store with intensity
        for (idx, value) in peaks(time_slice)[:n_peaks]:
            hz = bin2freq[idx] 
            if hz not in pitches:
                pitches.append(hz)
                intensities.append(value)

        # account for not enough peaks (silence)
        while len(pitches) < n_peaks:
            pitches.append(1)
            intensities.append(ZERO_VOLUME)

        top_freqs.append((pitches, intensities))
    return top_freqs 


def write_stream(path, s):
    
    s.insert(0, tempo.MetronomeMark(number=1500))
   
    s.write("midi", path)

def generate_midi(data, sample_rate, output_file, params):
    
    
    spec = librosa.stft(data.T[0], n_fft=4096, hop_length=512)
    
    
    db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    
    
    
   
    top_freqs = compute_top_frequencies(db, params['n_peaks'])
    s = make_stream(top_freqs, params['keydiff_threshold'])
    
    write_stream(output_file, s)
    


def wav2midi(input_wav, output_midi, params):
    data, sample_rate = sf.read(input_wav, dtype='float32')
    generate_midi(data, sample_rate, output_midi, params)

def main():
    
    wav2midi(r"C:\Users\ache\input0.wav", r"C:\Users\ache\output0.mid", {'n_peaks': 12, 'keydiff_threshold': 1})
    


if __name__ == "__main__":
    main()

