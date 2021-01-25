from pathlib import Path

import markdown
import pandas as pd

AUDIO_PATH = Path('/Users/gfodor/kaggle/rainforest/rfcx/cache/train3')


def get_candidates():
    return pd.read_csv('./data/candidates.csv')


def show_references():
    with open('rainforest/ref.md', 'r', encoding='utf-8') as f:
        text = f.read()
    html = markdown.markdown(text)
    return html
