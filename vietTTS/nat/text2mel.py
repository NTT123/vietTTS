import pickle
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import FLAGS, DurationInput
from .data_loader import load_phonemes_set_from_lexicon_file
from .model import AcousticModel, DurationModel


def load_lexicon(fn):
    lines = open(fn, "r").readlines()
    lines = [l.lower().strip().split("\t") for l in lines]
    return dict(lines)


def predict_duration(tokens):
    def fwd_(x):
        return DurationModel(is_training=False)(x)

    forward_fn = jax.jit(hk.transform_with_state(fwd_).apply)
    with open(FLAGS.ckpt_dir / "duration_latest_ckpt.pickle", "rb") as f:
        dic = pickle.load(f)
    x = DurationInput(
        np.array(tokens, dtype=np.int32)[None, :],
        np.array([len(tokens)], dtype=np.int32),
        None,
    )
    return forward_fn(dic["params"], dic["aux"], dic["rng"], x)[0]


consonants = (
    ["ngh"]
    + ["ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "tr", "th"]
    + ["b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"]
)
vowels = (
    ["a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y"]
    + ["á", "ắ", "ấ", "é", "ế", "í", "ó", "ố", "ớ", "ú", "ứ", "ý"]
    + ["à", "ằ", "ầ", "è", "ề", "ì", "ò", "ồ", "ờ", "ù", "ừ", "ỳ"]
    + ["ả", "ẳ", "ẩ", "ẻ", "ể", "ỉ", "ỏ", "ổ", "ở", "ủ", "ử", "ỷ"]
    + ["ã", "ẵ", "ẫ", "ẽ", "ễ", "ĩ", "õ", "ỗ", "ỡ", "ũ", "ữ", "ỹ"]
    + ["ạ", "ặ", "ậ", "ẹ", "ệ", "ị", "ọ", "ộ", "ợ", "ụ", "ự", "ỵ"]
)

phonemes = consonants + vowels


def merge_vowels(phones):
    out = []
    for ph in phones:
        if ph in vowels:
            if len(out) > 0 and out[-1][0] in vowels:
                out[-1] = out[-1] + ph
            else:
                out.append(ph)
        else:
            out.append(ph)
    return out


def word_to_phonemes(word):
    word = unicodedata.normalize("NFKC", word.strip().lower())
    idx = 0
    out = []
    while idx < len(word):
        # length: 3, 2, 1
        for l in [3, 2, 1]:
            if idx + l <= len(word) and word[idx : (idx + l)] in phonemes:
                out.append(word[idx : (idx + l)])
                idx = idx + l
                break
        else:
            raise ValueError(f"Unknown phoneme {word[idx]}")
    out = merge_vowels(out)
    return out


def text2tokens(text, lexicon_fn):
    phonemes = load_phonemes_set_from_lexicon_file(lexicon_fn)
    lexicon = load_lexicon(lexicon_fn)

    words = text.strip().lower()
    phones = []
    for word in words:
        if word in lexicon:
            phones.extend(lexicon[word].split())
            phones.append(FLAGS.special_phonemes[FLAGS.word_end_index])
        elif word == FLAGS.special_phonemes[FLAGS.sil_index]:
            phones.append(FLAGS.special_phonemes[FLAGS.sil_index])
        else:
            phones.extend(word_to_phonemes(word))
            phones.append(FLAGS.special_phonemes[FLAGS.word_end_index])

    tokens = [FLAGS.sil_index]
    for phone in phones:
        tokens.append(phonemes.index(phone))
    tokens.append(FLAGS.sil_index)  # silence
    return tokens


def predict_mel(tokens, durations):
    ckpt_fn = FLAGS.ckpt_dir / "acoustic_latest_ckpt.pickle"
    with open(ckpt_fn, "rb") as f:
        dic = pickle.load(f)
        last_step, params, aux, rng, optim_state = (
            dic["step"],
            dic["params"],
            dic["aux"],
            dic["rng"],
            dic["optim_state"],
        )

    @hk.transform_with_state
    def forward(tokens, durations, n_frames):
        net = AcousticModel(is_training=False)
        return net.inference(tokens, durations, n_frames)

    durations = durations * FLAGS.sample_rate / (FLAGS.n_fft // 4)
    n_frames = int(jnp.sum(durations).item())
    predict_fn = jax.jit(forward.apply, static_argnums=[5])
    tokens = np.array(tokens, dtype=np.int32)[None, :]
    return predict_fn(params, aux, rng, tokens, durations, n_frames)[0]


def text2mel(
    text: str, lexicon_fn=FLAGS.data_dir / "lexicon.txt", silence_duration: float = -1.0
):
    tokens = text2tokens(text, lexicon_fn)
    durations = predict_duration(tokens)
    durations = jnp.where(
        np.array(tokens)[None, :] == FLAGS.sil_index,
        jnp.clip(durations, a_min=silence_duration, a_max=None),
        durations,
    )
    durations = jnp.where(
        np.array(tokens)[None, :] == FLAGS.word_end_index, 0.0, durations
    )
    mels = predict_mel(tokens, durations)
    if tokens[-1] == FLAGS.sil_index:
        end_silence = durations[0, -1].item()
        silence_frame = int(end_silence * FLAGS.sample_rate / (FLAGS.n_fft // 4))
        mels = mels[:, : (mels.shape[1] - silence_frame)]
    return mels


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mel = text2mel(args.text)
    plt.figure(figsize=(10, 5))
    plt.imshow(mel[0].T, origin="lower", aspect="auto")
    plt.savefig(str(args.output))
    plt.close()
    mel = jax.device_get(mel)
    mel.tofile("clip.mel")
