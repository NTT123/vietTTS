import re
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf

from .hifigan.mel2wave import mel2wave
from .nat.config import FLAGS
from .nat.text2mel import text2mel

parser = ArgumentParser()
parser.add_argument('--text', type=str)
parser.add_argument('--output', default='clip.wav', type=Path)
parser.add_argument('--sample-rate', default=16000, type=int)
parser.add_argument('--silence-duration', default=-1, type=float)
parser.add_argument('--lexicon-file', default=None)
parser.add_argument('--use-waveRNN', default=False, action='store_true')
args = parser.parse_args()


def nat_normalize_text(text):
  text = unicodedata.normalize('NFKC', text)
  text = text.lower().strip()
  sp = FLAGS.special_phonemes[FLAGS.sp_index]
  text = re.sub(r'[\n.,:]+', f' {sp} ', text)
  text = text.replace('"', " ")
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'[.,:;?!]+', f' {sp} ', text)
  text = re.sub('[ ]+', ' ', text)
  text = re.sub(f'( {sp}+)+ ', f' {sp} ', text)
  return text.strip()


text = nat_normalize_text(args.text)
print('Normalized text input:', text)
mel = text2mel(text, args.lexicon_file, args.silence_duration)
if args.use_waveRNN:
  from .waveRNN.mel2wave import mel2wave
  wave = mel2wave(mel)
else:
  wave = mel2wave(mel)
print('writing output to file', args.output)
sf.write(str(args.output), wave, samplerate=args.sample_rate)
