#!/usr/bin/env bash

# shellcheck disable=SC2164
cd /home/dhy/work/CDFSL/filelists

# convert models
cd /home/dhy/work/CDFSL/filelists/miniImagenet
python write_miniImagenet_filelist.py

cd /home/dhy/work/CDFSL/filelists/cub
python write_cub_filelist.py

cd /home/dhy/work/CDFSL/filelists/cars
python write_cars_filelist.py

cd /home/dhy/work/CDFSL/filelists/places
python write_places_filelist.py

cd /home/dhy/work/CDFSL/filelists/plantae
python write_plantae_filelist.py

cd /home/dhy/work/CDFSL/filelists/CropDiseases
python write_CropDiseases_filelist.py

cd /home/dhy/work/CDFSL/filelists/EuroSAT
python write_EuroSAT_filelist.py

cd /home/dhy/work/CDFSL/filelists/ISIC
python write_ISIC_filelist.py