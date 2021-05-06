A Vietnamese TTS
================

Tacotron + WaveRNN for vietnamese datasets.

A synthesized audio clip is at [assets/reinfo/clip.wav](assets/reinfo/clip.wav).

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


Download reinfo dataset
-----------------------

```sh
bash ./scripts/download_reinfo_dataset.sh
```


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



Train waveRNN
-------------

```sh
python3 -m vietTTS.waveRNN.trainer
```


Synthesize speech
-----------------

```sh
python3 -m vietTTS.synthesizer --use-nat --text="hôm qua em tới trường" --output=clip.wav
```