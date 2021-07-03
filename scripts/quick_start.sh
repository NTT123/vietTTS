if [ ! -f assets/infore/hifigan/g_00710000 ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p assets/infore/{nat,hifigan}
  gdown --id 1zh6axidF1H9TAz9t45IrSe_mQ8KZMgNr -O assets/infore/nat/nat_ckpt_latest.pickle
  gdown --id 10dsJ7zO7qdUHGzAzWRa9fepHgaect4ey -O assets/infore/hifigan/g_00710000
  python3 -m vietTTS.hifigan.convert_torch_model_to_haiku --config-file=assets/hifigan/config.json --checkpoint-file=assets/infore/hifigan/g_00710000
fi

echo "Generate audio clip"
text=`cat assets/transcript.txt`
python3 -m vietTTS.synthesizer --speaker 0 --text "$text" --output assets/infore/clip.wav --lexicon-file assets/infore/lexicon.txt --silence-duration 0.2