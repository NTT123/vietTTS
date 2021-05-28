data_root="./train_data" # modify this
pushd .
mkdir -p $data_root
cd $data_root
gdown --id 1Pe-5lKT_lZsliv2WxQDai2mjhI9ZMFlj -O infore.zip
unzip -q infore.zip 
popd
