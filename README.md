# Another TTS 

### Install

```sh
git clone https://github.com/NTT123/vietTTS.git
cd vietTTS 
pip3 install -e .
```

### Download reinfo dataset

```sh
bash ./scripts/download_reinfo_dataset.sh
```

### Train Tacotron 

```sh
python3 -m vietTTS.tacotron.trainer
```

### Train waveRNN

```sh
python3 -m vietTTS.waveRNN.trainer
```


### Synthesize speech

```sh
python3 -m vietTTS.synthesizer -text="hôm qua em tới trường" --output=clip.wav
```