if [ ! -f assets/infore/hifigan/g_00305000 ]; then
  pip3 install gdown
  echo "Downloading models..."
  mkdir -p -p assets/infore/{nat,hifigan}
  gdown --id 15rEp52SIxUdPlr4sSsr1YrzIcL-XnRr3 -O assets/infore/nat/duration_ckpt_latest.pickle
  gdown --id 188u0vv-v9vB6CsrCiHzzeyqC66gXuQtZ -O assets/infore/nat/acoustic_ckpt_latest.pickle
  gdown --id 1-FKSz8aQ9MFlA2nhFd__7JXjGw5QLskf -O assets/infore/hifigan/g_00305000
  python3 -m vietTTS.hifigan.convert_torch_model_to_haiku --config-file=assets/hifigan/config.json --checkpoint-file=assets/infore/hifigan/g_00305000
fi

echo "Generate audio clip"
text=`cat assets/transcript.txt`
python3 -m vietTTS.synthesizer --text "$text" --output assets/infore/clip.wav --use-hifigan --use-nat --lexicon-file assets/infore/lexicon.txt --silence-duration 0.2