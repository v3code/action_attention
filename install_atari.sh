#!/bin/bash
wget https://github.com/greydanus/baby-a3c/raw/master/pong-v4/model.80.tar -P ./model/

pip install -r requirements.txt
pip install gym[atari]
pip install autorom[accept-rom-license]
mkdir ../Roms
AutoROM --accept-license --install-dir ../Roms
python -m atari_py.import_roms ../Roms