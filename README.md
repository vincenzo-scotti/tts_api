# Mozilla TTS API

API for [Mozilla TTS library](https://github.com/mozilla/TTS).
This repository contains an installation guide and some utility functions to simplify the access to the models in this library.
All credits go to the original developers.

## Repository structure

This repository is organised into two main directories:

- `resources/` contains:
    - directories to host the TTS models;
    - directory to host the vocoder models;
    - directory to host the speaker embedding model.
- `tts_api/` package with the api.

For further details on the available models, refer to the `README.md` in the `resources/` directory.

## Environment

To install all the required packages within an anaconda environment, run the following commands:

```bash
# Create anaconda environment (skip cudatoolkit option if you don't want to use the GPU)
conda create -n ttsapi python=3.10 cudatoolkit=11.3
# Activate anaconda environment
conda activate ttsapi
# Install packages
conda install pytorch=1.11.0 -c pytorch
conda install -c conda-forge numpy=1.21 tensorboard=2.9.1 pandas scikit-learn librosa matplotlib seaborn jupyterlab
# Repo for Text-to-Speech with its requirements 
git clone https://github.com/mozilla/TTS -b dev
cd TTS
# See https://discourse.mozilla.org/t/size-mismatches-in-released-model/75538/3
git checkout ea976b0  
conda install -c conda-forge unidecode attrdict tensorboardx pysoundfile pysbd pyworld pydub inflect=5.6.0
pip install "phonemizer>=2.2.0"
cd ..
# Additional repo for the Speaker Encoder with its requirements
git clone https://github.com/Edresson/GE2E-Speaker-Encoder.git -b speaker-encoder
cd GE2E-Speaker-Encoder
sed -i '.tmp' 's/checkpoint = torch.load(weights_fpath)/checkpoint = torch.load(weights_fpath, map_location=_device)/' encoder/inference.py
conda install -c conda-forge umap-learn visdom webrtcvad
# pip install umap-learn visdom webrtcvad "librosa>=0.5.1" "matplotlib>=2.0.2"
cd ..
```

Notice that it is necessary to clone separately the TTS and speaker encoder repository, this additional repositories must be on path.
Moreover, the TTS repository must be rebased to a specific commit and branch.

## Example

Here follows a usage example:
```python
import torch
from tts_api import load_tts, load_vocoder, load_speaker_encoder, synthesise_speech


# Reference audio for voice (optional)
audio_path = 'path/to/audio.wav'
# Load model instances
tts, tts_configs, tts_ap = load_tts(
    'resources/models/tts/tacotron_2_multispeaker/config.json', 
    'resources/models/tts/tacotron_2_multispeaker/checkpoint_220000.pth.tar', 
    tts_model_scale_stats_path='resources/models/tts/tacotron_2_multispeaker/scale_stats.npy',
    tts_model_speaker_file='resources/models/tts/tacotron_2_multispeaker/speakers.json'
)
vocoder, vocoder_configs, vocoder_ap = load_vocoder(
    'resources/models/vocoder/fb_melgan/config.json', 
    'resources/models/vocoder/fb_melgan/best_model.pth.tar'
)
load_speaker_encoder('resources/models/speaker_encoder/ge2e/pretrained.pt')

# Syntehsise speech
synthesise_speech(
    "I am testing a neural network for speech synthesis.", 
    tts, 
    tts_configs, 
    tts_ap, 
    vocoder_model=vocoder,
    vocoder_configs=vocoder_configs,
    vocoder_ap=vocoder_ap,
    speaker_reference_clip_path=audio_path, 
    device=torch.device('cpu'), 
    out_path='path/to/output.wav'
)
```
