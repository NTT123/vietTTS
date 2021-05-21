A Vietnamese TTS
================

Tacotron + HiFiGAN vocoder for vietnamese datasets.

A synthesized audio clip: [clip.wav](assets/infore/clip.wav). A colab notebook: [notebook](https://colab.research.google.com/drive/1oczrWOQOr1Y_qLdgis1twSlNZlfPVXoY?usp=sharing).

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


Download InfoRe dataset
-----------------------

```sh
bash ./scripts/download_aligned_infore_dataset.sh
```

**Note**: this is a denoised and aligned version of the original dataset which is donated by the InfoRe Technology company (see [here](https://www.facebook.com/groups/j2team.community/permalink/1010834009248719/)). You can download the original dataset (**InfoRe Technology 1**) at [here](https://github.com/TensorSpeech/TensorFlowASR/blob/main/README.md#vietnamese).


Train duration model
--------------------

```sh
python3 -m vietTTS.nat.duration_trainer
```


Train acoustic model
--------------------
```sh
python3 -m vietTTS.nat.acoustic_trainer
```



Train HiFiGAN vocoder
-------------

We use the original implementation from HiFiGAN authors at https://github.com/jik876/hifi-gan. Use the config file at `assets/hifigan/config.json` to train your model.

```sh
git clone https://github.com/jik876/hifi-gan.git

# create dataset in hifi-gan format
ln -sf `pwd`/train_data hifi-gan/data
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
  --text="hôm qua em tới trường" \
  --output=clip.wav
```
