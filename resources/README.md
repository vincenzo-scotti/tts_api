# Models

This directory is used to host the pre-trained models (TTS, vocoder, and speaker encoder).
All credits to the original authors and contributors that trained the models (see links below).

Voice synthesis:
- Spectrogram generation:
  - [Tacotron 2](http://proceedings.mlr.press/v80/wang18h.html):
    - [w/ speaker embeddings](https://proceedings.neurips.cc/paper/2018/hash/6832a7b24bc06775d02b7406880b93fc-Abstract.html) ([weights checkpoint](https://drive.google.com/uc?id=1LOp9NMpxQzobRiQkEq32B11_uu6g06Ma), [configuration file](https://drive.google.com/uc?id=1RuS5YEX0_DWPQo7Od_o7_9YGBBflDM1w), [speaker embeddings](https://drive.google.com/uc?id=1AZAWxW67MRgKNTeYvWcLBN_RN-D8B-iJ), [scaling statistics](https://drive.google.com/uc?id=1BHmVmi4gTLZE5ITE9EwCTNJT2x-9Uygj));
    - [w/ GST and speaker embeddings](http://proceedings.mlr.press/v80/skerry-ryan18a.html) (GSTs are [Global Style Tokens](http://proceedings.mlr.press/v80/wang18h.html)):
      - Separate embeddings for style and speaker ([weights checkpoint, configuration file and speaker embeddings file](https://github.com/Edresson/TTS/releases/download/v1.0.0/Checkpoints-TTS-MultiSpeaker-Jia-et-al-2018-with-GST-CorentinJ_SpeakerEncoder_and_DDC.zip));
      - Combined embeddings for style and speaker ([weights checkpoint](https://drive.google.com/uc?id=1iDCL_cRIipoig7Wvlx4dHaOrmpTQxuhT), [configuration file](https://drive.google.com/uc?id=1YKrAQKBLVXzyYS0CQcLRW_5eGfMOIQ-2), [speaker embeddings](https://drive.google.com/uc?id=1oOnPWI_ho3-UJs3LbGkec2EZ0TtEOc_6)).
- Vocoder:
  - [Full-Band MelGAN](https://doi.org/10.1109/SLT48900.2021.9383551) ([weights checkpoint](https://drive.google.com/file/d/1K3KBl3rxngIaOIBI7ujvDcAQn1RQhkhA), [configuration file](https://drive.google.com/file/d/1uBRVNxsoCYJxNCqPoQedASm6EtSW3w04/), [scaling statistics](https://drive.google.com/file/d/1O8ziB27XqzIpkb-6_QI0fpDouF4-v7_1))
  - [WaveGrad](https://openreview.net/forum?id=NsMLjcFaO8O) ([weights checkpoint](https://drive.google.com/uc?id=1r2g90JaZsfCj9dJkI9ioIU6JCFMPRqi6), [configuration file](https://drive.google.com/uc?id=1POrrLf5YEpZyjvWyMccj1nGCVc94mR6s), [scaling statistics](https://drive.google.com/uc?id=1Vwbv4t-N1i3jXqI0bgKAhShAEO097sK0)).
- Speaker encoder
  - [GE2E](https://ieeexplore.ieee.org/abstract/document/8462665) for [speaker conditioning](https://proceedings.neurips.cc/paper/2018/hash/6832a7b24bc06775d02b7406880b93fc-Abstract.html) ([weights checkpoint](https://github.com/Edresson/Real-Time-Voice-Cloning/releases/download/checkpoints/pretrained.zip)).

Please refer to the [Conqui TTS](https://github.com/coqui-ai/TTS) repository by [Conqui.ai](https://coqui.ai) (formerly [TTS: Text-to-Speech for all](https://github.com/mozilla/TTS) by [Mozilla](https://www.mozilla.org/)) for further details on speech synthesis models and to the [GE2E Speaker Encoder](https://github.com/Edresson/GE2E-Speaker-Encoder) repository by [Edresson Casanova](https://www.linkedin.com/in/edresson/).
For simplicity, we provide a separate zip file with all the model checkpoints necessary to speech synthesis ([link](https://polimi365-my.sharepoint.com/:u:/g/personal/10451445_polimi_it/EdcWpb1EEh9KsPG_hkVEkboBfNNNxk4BKJc_B1rrlsbWJQ?e=3x62Y6)).

Directory structure:
```
 |- resources/
    |- speaker_encoder/
      |- ge2e
        |- pretrained.pt
    |- tts/
      |- tacotron_2_gst_multispeaker
        |- best_model.pth.tar
        |- config.json
        |- speakers.json
      |- tacotron_2_gst_multispeaker_combined
        |- best_model.pth.tar
        |- config.json
        |- speakers.json
      |- tacotron_2_multispeaker
        |- checkpoint_220000.pth.tar
        |- config.json
        |- scale_stats.npy
        |- speakers.json
    |- vocoder/
      |- fb_melgan
        |- best_model.pth.tar
        |- config.json
        |- scale_stats.npy
      |- wavegrad
        |- checkpoint_345000.pth.tar
        |- config.json
        |- scale_stats_wavegrad.npy
```

Notes: 
- TTSes were trained on 22050 Hz audio clips, Vocoders were trained on 24000 Hz audio clips, there is an issue to help use model ([link](https://github.com/mozilla/TTS/issues/520)).
- Vocoder scaling stats should not be used in case of multi speaker model according to model config file comments.
- Only the checkpoint of the speaker encoder can be retained from the corresponding archive, it is located in `pretrained/encoder/saved_models/`
- To have the speaker encoder work out of the box it is necessary to change line 33 in file `./GE2E-Speaker-Encoder/encoder/inference.py` from  `checkpoint = torch.load(weights_fpath)` to `checkpoint = torch.load(weights_fpath, map_location=_device)`.
- Other info on the pre-trained models can be found at the following [link](https://github.com/mozilla/TTS/wiki/Released-Modelss).
- Detailed examples of usage of Tacotron 2 with GST and speaker embeddings at the following [link](https://colab.research.google.com/drive/1Gtt9EV1fFzuKbOdqUrLuAMuxBaot5v4F?usp=sharing#scrollTo=UmftUXTRLYEx)
