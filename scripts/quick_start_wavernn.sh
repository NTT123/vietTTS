if [ ! -f assets/infore/waveRNN/checkpoint_00310000.pickle ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p -p assets/infore/{nat,waveRNN}
  gdown --id 16UhN8QBxG1YYwUh8smdEeVnKo9qZhvZj -O assets/infore/nat/duration_ckpt_latest.pickle
  gdown --id 1-8Ig65S3irNHSzcskT37SLgeyuUhjKdj -O assets/infore/nat/acoustic_ckpt_latest.pickle
  gdown --id 1-tJfXqaHOKsbioYyACIs4eRRaJP0qLWZ -O assets/infore/waveRNN/checkpoint_00310000.pickle
fi

echo "Generate audio clip"
text=`cat assets/transcript.txt`
python3 -m vietTTS.synthesizer --text "$text" --use-waveRNN --output assets/infore/clip.wav --lexicon-file assets/infore/lexicon.txt --silence-duration 0.2