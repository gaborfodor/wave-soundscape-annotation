import pandas as pd
import soundfile as sf
from tqdm import tqdm

from rainforest.config import SAMPLE_RATE
from rainforest.utils import get_candidates, get_chunk, WAV_CACHE_PATH, get_original_tp, get_original_fp

WAV_CACHE_PATH.mkdir(parents=True, exist_ok=True)

df = get_candidates()

for rec_id, start in tqdm(df[['rec_id', 'start']].values):
    wav_path = WAV_CACHE_PATH / f'{rec_id}_{start}.wav'
    if not wav_path.exists():
        y = get_chunk(rec_id, start)
        sf.write(wav_path, y, SAMPLE_RATE)

df = pd.concat([
    get_original_tp(),
    get_original_fp()
])
df['start'] = df['t_min'].values - 0.5
for rec_id, start in tqdm(df[['recording_id', 'start']].values):
    start = int(start)
    wav_path = WAV_CACHE_PATH / f'{rec_id}_{start}.wav'
    if not wav_path.exists():
        try:
            y = get_chunk(rec_id, start)
            sf.write(wav_path, y, SAMPLE_RATE)
        except FileNotFoundError as e:
            print(e)


print(df.head())
