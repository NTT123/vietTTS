A Vietnamese TTS
================

Non-Attentive Tacotron (NAT) + HiFiGAN vocoder for vietnamese datasets. Read NAT paper at [here](https://arxiv.org/abs/2010.04301).

A synthesized audio clip: [clip.wav](assets/infore/clip.wav).

Install
-------


```sh
git clone https://github.com/NTT123/vietTTS.git
cd vietTTS 
pip3 install -e .
```


Quick start using pretrained models
----------------------------------
```sh
bash ./scripts/quick_start.sh
```


Download VIVOS+InfoRe dataset
-----------------------

```sh
bash ./scripts/download_aligned_vivos_infore_dataset.sh
```

**Note**: This dataset is created from two public datasets:
- **VIVOS dataset** (15 hours, 65 speakers) from HCMUS university. You can download the original dataset at [here](https://ailab.hcmus.edu.vn/vivos). 
- **Infore dataset** (25 hours, 1 speaker) is donated by the InfoRe Technology company (see [here](https://www.facebook.com/groups/j2team.community/permalink/1010834009248719/)). You can download the original InfoRe dataset (**InfoRe Technology 1**) at [here](https://github.com/TensorSpeech/TensorFlowASR/blob/main/README.md#vietnamese).

The Montreal Forced Aligner (MFA) is used to align transcript and speech (textgrid files). [Here](https://colab.research.google.com/gist/NTT123/95b12ca42a4bdd1a856aba0fbb0f8936/infore-mfa-tutorial.ipynb) is a Colab notebook to align InfoRe dataset. Visit [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/) for more information on how to create textgrid files.

Train acoustic model
--------------------

```sh
python3 -m vietTTS.nat.trainer
```




Train HiFiGAN vocoder
-------------

We use the original implementation from HiFiGAN authors at https://github.com/jik876/hifi-gan. Use the config file at `assets/hifigan/config.json` to train your model.

```sh
python3 -m vietTTS.nat.zero_silence_segments -o hifigan_train_data # zero all [sil, sp, spn] segments
git clone https://github.com/jik876/hifi-gan.git

# create dataset in hifi-gan format
ln -sf `pwd`/hifigan_train_data hifi-gan/data
cd hifi-gan/data
ls -1 *.wav | sed -e 's/\.wav$//' > files.txt
cd ..
head -n 100 data/files.txt > val_files.txt
tail -n +101 data/files.txt > train_files.txt
rm data/files.txt

# training
python3 train.py \
  --config ../assets/hifigan/config.json \
  --input_wavs_dir=data \
  --input_training_file=train_files.txt \
  --input_validation_file=val_files.txt
```


Then, use the following command to convert pytorch model to haiku format:
```sh
cd ..
python3 -m vietTTS.hifigan.convert_torch_model_to_haiku \
  --config-file=assets/hifigan/config.json \
  --checkpoint-file=hifi-gan/cp_hifigan/g_[latest_checkpoint]
```

Synthesize speech
-----------------

```sh
python3 -m vietTTS.synthesizer \
  --lexicon-file=train_data/lexicon.txt \
  --speaker=0 \
  --text="hôm qua em tới trường" \
  --output=clip.wav
```
