from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf

from .hifigan.mel2wave import mel2wave
from .nat.text2mel import text2mel

parser = ArgumentParser()
parser.add_argument('--text', type=str)
parser.add_argument('--output', default='clip.wav', type=Path)
parser.add_argument('--sample-rate', default=16000, type=int)
parser.add_argument('--silence-duration', default=-1, type=float)
parser.add_argument('--lexicon-file', default=None)
args = parser.parse_args()


def nat_normalize_text(text):
  text = text.lower().strip()
  import re
  text = re.sub(r'[\n.,:]+', ' sp ', text)
  text = re.sub('[ ]+', ' ', text)
  return text.strip()


text = nat_normalize_text(args.text)
print('Normalized text input:', text)
mel = text2mel(text, args.lexicon_file, args.silence_duration)
wave = mel2wave(mel)
print('writing output to file', args.output)
sf.write(str(args.output), wave, samplerate=args.sample_rate)
