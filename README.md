# Soundsense
 Acoustic Anomaly Detection for Predictive Maintenance 

## Description 
SoundSense detets anomalies in industrial machine sounds using mel-spectograms and a CNN classifier that enables predictive maintenance before failures occur.

##Dataset 
MIMII Dataset 
- Valve machine type (id_00, id_02, id_04, id_06)
- 1110 audio clips | 10 s per clip | 16kHz
- Normal vs Abnormal sounds

## Installation 
pip install librosa matplotlib numpy jupyter soundfile

## Project Structure 
├── data/            # MIMI dataset(not tracked by git)

├── notebooks/       # EDA and experiments

|    └── eda.ipynb

└── src/             # Source code (coming)
