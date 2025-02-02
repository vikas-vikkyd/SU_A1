# Classify the Urban Sound

# Author

- Vikas Kumar Singh - M23AIR545

## Steps
- Run all the commands from from project directory
- Create a virtual env using `python -m venv env` (Python 3.10 is recommended)
- Activate the virtual env using `source env/bin/activate`
- Use `pip install -r requirements.txt` to install the packages
- Use `python get_spectogram.py` to generate spectrogram using different window and extract features from spectrogram which will be used to classify the sound
- Use `python train_cls.py` to train the model

# References
+ https://keras.io/examples/vision/pointnet/#build-a-model