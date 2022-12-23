#!/bin/bash
pip install -r requirements.txt
pip install gym[atari]
pip install autorom[accept-rom-license]
mkdir ../Roms
AutoROM --accept-license --install-dir ../Roms
python -m atari_py.import_roms ../Roms