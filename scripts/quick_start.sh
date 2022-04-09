if [ ! -f assets/infore/hifigan/g_00800000 ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p -p assets/infore/{nat,hifigan}
  gdown 16UhN8QBxG1YYwUh8smdEeVnKo9qZhvZj -O assets/infore/nat/duration_latest_ckpt.pickle
  gdown 1-8Ig65S3irNHSzcskT37SLgeyuUhjKdj -O assets/infore/nat/acoustic_latest_ckpt.pickle
  gdown 10SFOlAduG20TdjGC5e1Jod_vJIpxkD6u -O assets/infore/hifigan/g_00800000
  python3 -m vietTTS.hifigan.convert_torch_model_to_haiku --config-file=assets/hifigan/config.json --checkpoint-file=assets/infore/hifigan/g_00800000
fi

echo "Generate audio clip"
text=`cat assets/transcript.txt`
python3 -m vietTTS.synthesizer --text "$text" --output assets/infore/clip.wav --lexicon-file assets/infore/lexicon.txt --silence-duration 0.2