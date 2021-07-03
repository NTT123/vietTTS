data_root="./train_data" # modify this
pushd .
mkdir -p $data_root
cd $data_root
gdown --id 1srTt_c_a4ed1yq8xujixyYZAkXRp5a0F -O vivos_infore.zip
unzip -q vivos_infore.zip 
popd
