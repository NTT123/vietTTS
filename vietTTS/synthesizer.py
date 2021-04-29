from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--text', type=str)
parser.add_argument('--output', default='clip.wav', type=Path)
args = parser.parse_args()

raise NotImplementedError()
