A Vietnamese TTS
================

Duration model + Acoustic model + HiFiGAN vocoder for vietnamese text-to-speech application.

Online demo at https://huggingface.co/spaces/ntt123/vietTTS.

A synthesized audio clip: [clip.wav](assets/infore/clip.wav). A colab notebook: [notebook](https://colab.research.google.com/drive/1oczrWOQOr1Y_qLdgis1twSlNZlfPVXoY?usp=sharing).


🔔Checkout the experimental `multi-speaker` branch (`git checkout multi-speaker`) for multi-speaker support.🔔

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
python ./scripts/download_aligned_infore_dataset.py
```

**Note**: this is a denoised and aligned version of the original dataset which is donated by the InfoRe Technology company (see [here](https://www.facebook.com/groups/j2team.community/permalink/1010834009248719/)). You can download the original dataset (**InfoRe Technology 1**) at [here](https://github.com/TensorSpeech/TensorFlowASR/blob/main/README.md#vietnamese).

See `notebooks/denoise_infore_dataset.ipynb` for instructions on how to denoise the dataset. We use the Montreal Forced Aligner (MFA) to align transcript and speech (textgrid files). 
See `notebooks/align_text_audio_infore_mfa.ipynb` for instructions on how to create textgrid files.

Train duration model
--------------------

```sh
python -m vietTTS.nat.duration_trainer
```


Train acoustic model
--------------------
```sh
python -m vietTTS.nat.acoustic_trainer
```



Train HiFiGAN vocoder
-------------

We use the original implementation from HiFiGAN authors at https://github.com/jik876/hifi-gan. Use the config file at `assets/hifigan/config.json` to train your model.

```sh
git clone https://github.com/jik876/hifi-gan.git

# create dataset in hifi-gan format
ln -sf `pwd`/train_data hifi-gan/data
cd hifi-gan/data
ls -1 *.TextGrid | sed -e 's/\.TextGrid$//' > files.txt
cd ..
head -n 100 data/files.txt > val_files.txt
tail -n +101 data/files.txt > train_files.txt
rm data/files.txt

# training
python train.py \
  --config ../assets/hifigan/config.json \
  --input_wavs_dir=data \
  --input_training_file=train_files.txt \
  --input_validation_file=val_files.txt
```

Finetune on Ground-Truth Aligned melspectrograms:
```sh
cd /path/to/vietTTS # go to vietTTS directory
python -m vietTTS.nat.zero_silence_segments -o train_data # zero all [sil, sp, spn] segments
python -m vietTTS.nat.gta -o /path/to/hifi-gan/ft_dataset  # create gta melspectrograms at hifi-gan/ft_dataset directory

# turn on finetune
cd /path/to/hifi-gan
python train.py \
  --fine_tuning True \
  --config ../assets/hifigan/config.json \
  --input_wavs_dir=data \
  --input_training_file=train_files.txt \
  --input_validation_file=val_files.txt
```

Then, use the following command to convert pytorch model to haiku format:
```sh
cd ..
python -m vietTTS.hifigan.convert_torch_model_to_haiku \
  --config-file=assets/hifigan/config.json \
  --checkpoint-file=hifi-gan/cp_hifigan/g_[latest_checkpoint]
```

Synthesize speech
-----------------

```sh
python -m vietTTS.synthesizer \
  --lexicon-file=train_data/lexicon.txt \
  --text="hôm qua em tới trường" \
  --output=clip.wav
```
