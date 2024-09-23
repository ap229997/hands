mkdir -p downloads
cd downloads

gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT # HaMeR
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
rm hamer_demo_data.tar.gz

gdown https://drive.google.com/drive/folders/1vjqBPicZagi0Xx0c_ItzAlISaj-pCigG --folder # WildHands
mv wildhands/example_data ./

cd ..