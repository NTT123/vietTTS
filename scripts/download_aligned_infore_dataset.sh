data_root="./train_data" # modify this
pushd .
mkdir -p $data_root
cd $data_root
gdown --id 1-6WuhWW8RXHswp7RNj06Z7KjU2XZmfQM -O infore.zip
unzip -q infore.zip 
popd
