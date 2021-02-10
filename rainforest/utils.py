import base64
import io
from pathlib import Path

import librosa
import librosa.display
import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

from rainforest.config import HOP_LENGTH, SAMPLE_RATE, N_FFT, CHUNK_SIZE, SPEC_FRANGE, N_MELS, FMAX, FMIN

AUDIO_PATH = Path('/Users/gfodor/kaggle/rainforest/rfcx/cache/train3_best')
DATA_PATH = Path('data')
TMP_PATH = Path('tmp')
ANNOTATION_PATH = Path('annotations')
WAV_CACHE_PATH = DATA_PATH / 'audio'

CANDIDATES_NAME = 'candidates.csv'


AUDIO_TEMPLATE = '''
<html>
    <body>
        <h4>{header}</h4>
        <audio controls src="{uploaded_audio}">
            Your browser does not support the <code>audio</code> element.
        </audio>
        <img src="{uploaded_image}" width="540" height="720">
    </body>
</html>
'''

def get_candidates():
    return pd.read_csv(DATA_PATH / CANDIDATES_NAME)


def get_original_tp():
    return pd.read_csv(DATA_PATH / 'train_tp.csv')


def get_original_fp():
    return pd.read_csv(DATA_PATH / 'train_fp.csv')


def get_annotations(filename):
    ann_path = ANNOTATION_PATH / filename
    if ann_path.exists():
        return pd.read_csv(ann_path)
    else:
        return pd.DataFrame(columns=['rec_id', 'start', 'spec_id', 'label'])


def get_next_candidate(candidates, annotations, spec_id, method, tp_selector):
    tp = 1 if tp_selector == 'TP' else 0
    c = candidates[candidates[spec_id] == tp].copy()
    c = c.merge(annotations[annotations.spec_id == spec_id], how='left', on=['rec_id', 'start'])
    c = c[c.label.isna()]

    if method == 'random':
        c['order'] = np.random.rand(len(c))
    else:
        c['order'] = c[f's{spec_id}']

    ascending = False if method == 'top' else True
    c = c.sort_values(by='order', ascending=ascending)
    return c[['rec_id', 'start', f's{spec_id}']].head(1).values[0]


def get_random_positive_example(spec_id):
    tp = get_original_tp()
    df = tp[(tp.species_id == int(spec_id)) & (tp.t_min < 58.5)].sample(n=1)
    rec_id = df['recording_id'].values[0]
    start = int(df['t_min'].values[0] - 0.5)
    return rec_id, start, 1.0


def get_random_negative_example(spec_id):
    fp = get_original_fp()
    df = fp[(fp.species_id == int(spec_id)) & (fp.t_min < 58.5)].sample(n=1)
    rec_id = df['recording_id'].values[0]
    start = int(df['t_min'].values[0] - 0.5)
    return rec_id, start, 0.0


def show_references():
    with open('rainforest/ref.md', 'r', encoding='utf-8') as f:
        text = f.read()
    html = markdown.markdown(text)
    return html


def get_chunk(rec_id, start):
    return np.load(AUDIO_PATH / f'{rec_id}_{start}.npy')


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image


def visualize_spectograms(recording_id, spec_id, start, p, show_boxes=True, fig_size=(8, 12)):
    original_tp = get_original_tp()
    original_fp = get_original_fp()
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX).round(0)
    y_ticks = np.arange(0, N_MELS, N_MELS // 8)
    y = get_chunk(recording_id, start)
    stft = np.abs(librosa.core.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH) ** 2)
    stft = librosa.power_to_db(stft, ref=np.max)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=fig_size)
    ax = axs[0]
    librosa.display.specshow(
        stft, hop_length=HOP_LENGTH, sr=SAMPLE_RATE, y_axis='hz', x_axis='time', cmap=plt.cm.Greys, ax=ax)
    ax.set_xlabel('')

    # Show original annotations
    if show_boxes:
        for _, row in original_tp[(original_tp.recording_id == recording_id)].iterrows():
            tmin = row['t_min']
            fmin = row['f_min']
            tmax = row['t_max']
            fmax = row['f_max']
            rect = patches.Rectangle(
                (tmin - start, fmin), tmax - tmin, fmax - fmin,
                linewidth=3, edgecolor='y', facecolor='none')
            ax.add_patch(rect)
        for _, row in original_fp[(original_fp.recording_id == recording_id)].iterrows():
            tmin = row['t_min']
            fmin = row['f_min']
            tmax = row['t_max']
            fmax = row['f_max']
            rect = patches.Rectangle(
                (tmin - start, fmin), tmax - tmin, fmax - fmin,
                linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    title = f'{recording_id} {spec_id} {p:.3} [{start}-{start + CHUNK_SIZE}]'
    ax.set_ylim(SPEC_FRANGE[spec_id])
    ax.grid()
    mel = librosa.feature.melspectrogram(
        y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, sr=SAMPLE_RATE, fmin=FMIN, fmax=FMAX)
    mel = librosa.power_to_db(mel, ref=np.max)
    axs[1].imshow(mel, cmap=plt.cm.Greys, origin='lower', interpolation='nearest', aspect='auto')

    fmin, fmax = SPEC_FRANGE[spec_id]
    imin = np.sum(mel_freqs <= fmin)
    imax = np.sum(mel_freqs <= fmax) + 1
    rect = patches.Rectangle((0, imin), mel.shape[1], imax - imin, linewidth=1, edgecolor='y', facecolor='none')
    axs[1].add_patch(rect)

    axs[1].grid()
    axs[1].set_ylabel('Mel scale')
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels([int(mel_freqs[i]) for i in y_ticks])
    plt.suptitle(title)
    plt.tight_layout()
    return fig
