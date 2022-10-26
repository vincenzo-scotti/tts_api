import random
import json

import numpy as np
import torch
import torch.nn.functional as F

from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.symbols import symbols, phonemes, make_symbols
from TTS.utils.io import load_config  # Config class loader
from TTS.tts.utils.generic_utils import setup_model  # TTS model setup
from TTS.tts.utils.io import load_checkpoint  # Model checkpoint loader
from TTS.vocoder.utils.generic_utils import setup_generator  # Vocoder model setup
from TTS.tts.utils.synthesis import synthesis  # Main wrapper for speech synthesis
from TTS.tts.utils.synthesis import compute_style_mel, numpy_to_torch, embedding_to_torch  # Utilities for GST encoding
from pathlib import Path
from encoder import inference as speaker_encoder_model
from encoder.params_model import model_embedding_size as speaker_embedding_size
from TTS.tts.utils.visual import visualize
import soundfile

from typing import Optional, Union, List, Dict, Tuple
from TTS.utils.io import AttrDict
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.vocoder.models.multiband_melgan_generator import MultibandMelganGenerator
from TTS.vocoder.models.fullband_melgan_generator import FullbandMelganGenerator
from TTS.vocoder.models.wavegrad import Wavegrad


def load_tts(
        tts_model_configs_path: str,
        tts_model_checkpoint_path: str,
        tts_model_scale_stats_path: Optional[str] = None,
        tts_model_speaker_file: Optional[str] = None,
        speaker_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        **config_kwargs
) -> Tuple[Tacotron2, AttrDict, AudioProcessor]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TTS configs
    tts_configs: AttrDict = load_config(tts_model_configs_path)
    tts_configs.forward_attn_mask = True
    if 'characters' in tts_configs.keys():
        symbols, phonemes = make_symbols(**tts_configs.characters)
    n_chars = len(phonemes) if tts_configs.use_phonemes else len(symbols)
    if tts_model_scale_stats_path is not None:
        tts_configs.audio['stats_path'] = tts_model_scale_stats_path
    if 'gst_use_speaker_embedding' not in tts_configs.gst:
        tts_configs.gst['gst_use_speaker_embedding'] = False
    # Audio Processor
    tts_ap = AudioProcessor(**tts_configs.audio)
    # Speaker configs  # TODO make embedding usable
    if tts_configs.use_external_speaker_embedding_file:
        speaker_mapping = json.load(open(tts_model_speaker_file, 'r'))
        n_speakers = len(speaker_mapping)
        speaker_file_id = list(speaker_mapping.keys())[
            speaker_idx if speaker_idx is not None else random.choice(range(n_speakers))
        ]
        speaker_embedding = speaker_mapping[speaker_file_id]['embedding']
    else:
        n_speakers = 1
    # Load TTS
    tts_model: Tacotron2 = setup_model(n_chars, n_speakers, tts_configs, speaker_embedding_dim=speaker_embedding_size)
    tts_model, _ = load_checkpoint(tts_model, tts_model_checkpoint_path, use_cuda=torch.cuda.is_available())
    # Move model to device and set in evaluation mode
    tts_model.to(device)
    tts_model.eval()

    return tts_model, tts_configs, tts_ap


def load_vocoder(
        vocoder_model_config_path: str,
        vocoder_model_checkpoint_path: str,
        vocoder_model_scale_stats_path: Optional[str] = None,
        device: Optional[torch.device] = None
) -> Tuple[Union[FullbandMelganGenerator, Wavegrad, MultibandMelganGenerator], AttrDict, AudioProcessor]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocoder_configs: AttrDict = load_config(vocoder_model_config_path)
    vocoder_configs.audio['stats_path'] = vocoder_model_scale_stats_path  # FIXME temporary solution

    vocoder_ap = AudioProcessor(**vocoder_configs.audio)

    vocoder_model: Union[FullbandMelganGenerator, Wavegrad, MultibandMelganGenerator] = setup_generator(vocoder_configs)
    vocoder_model.load_state_dict(torch.load(vocoder_model_checkpoint_path, map_location=torch.device('cpu'))['model'])
    if isinstance(vocoder_model, FullbandMelganGenerator):
        vocoder_model.remove_weight_norm()
    elif isinstance(vocoder_model, Wavegrad):
        vocoder_model.compute_noise_level(50, 1e-6, 1e-2)
    elif isinstance(vocoder_model, MultibandMelganGenerator):
        pass
    else:
        raise ValueError()
    vocoder_model.inference_padding = 0
    vocoder_model.to(device)
    vocoder_model.eval()

    return vocoder_model, vocoder_configs, vocoder_ap


def load_speaker_encoder(speaker_encoder_model_checkpoint_path: str, device: Optional[torch.device] = None):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speaker_encoder_model.load_model(Path(speaker_encoder_model_checkpoint_path), device=device.type)


@torch.no_grad()
def get_speaker_embedding(reference_audio_path: Union[str, List[str]]) -> List[float]:
    # Input normalisation
    if isinstance(reference_audio_path, str):
        return get_speaker_embedding([reference_audio_path])
    # Average the embeddings from the reference audio clips
    speaker_embedding: List[float] = np.vstack([
        speaker_encoder_model.embed_utterance(speaker_encoder_model.preprocess_wav(audio_file_path))
        for audio_file_path in reference_audio_path
    ]).mean(axis=0).tolist()

    return speaker_embedding


@torch.no_grad()
def get_prosody_embedding(
        reference_audio_path: str,
        tts_model: Tacotron2,
        tts_configs: AttrDict,
        tts_ap: Optional[AudioProcessor] = None,
        device: Optional[torch.device] = None
) -> List[float]:
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    # Extract Mel-Spectrogram of reference style speech
    style_mel_spectrogram: np.ndarray = compute_style_mel(reference_audio_path, tts_ap, cuda=use_cuda)
    style_mel_spectrogram: torch.tensor = numpy_to_torch(style_mel_spectrogram, torch.float, cuda=use_cuda)
    # Extract the prosody encoding
    prosody_embedding: torch.tensor = tts_model.gst_layer.encoder(style_mel_spectrogram).squeeze()
    # Get embedding on CPU if necessary
    if use_cuda:
        prosody_embedding = prosody_embedding.cpu()
    # Convert to float list
    prosody_embedding = prosody_embedding.tolist()

    return prosody_embedding


@torch.no_grad()
def get_gst_embedding_src_tgt(
        reference_audio_path: str,
        tts_model: Tacotron2,
        tts_configs: AttrDict,
        tts_ap: Optional[AudioProcessor] = None,
        device: Optional[torch.device] = None,

) -> Tuple[torch.tensor, torch.tensor]:  # Shapes ((batch, num. tokens, embedding dim.), (batch, embedding dim.))
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    # Extract Mel-Spectrogram of reference style speech
    style_mel_spectrogram: np.ndarray = compute_style_mel(reference_audio_path, tts_ap, cuda=use_cuda)
    style_mel_spectrogram: torch.tensor = numpy_to_torch(style_mel_spectrogram, torch.float, cuda=use_cuda)
    # Create dummy prosody embedding
    dummy_prosody_embedding = torch.zeros(1, tts_model.gst_embedding_dim // 2, device=device)
    # Extract token embeddings
    token_embeddings = tts_model.gst_layer.style_token_layer.style_tokens
    # Compute the separate embeddings to be combined using the style token amplifiers into the GST embedding
    if tts_configs.gst['gst_use_speaker_embedding']:
        # Extract speaker embeddings from reference speech
        speaker_embedding: List[float] = get_speaker_embedding(reference_audio_path)
        speaker_embedding: torch.tensor = embedding_to_torch(speaker_embedding, cuda=use_cuda)
        # Compute query
        query = torch.cat([dummy_prosody_embedding, speaker_embedding], dim=-1).reshape(1, 1, -1)
    else:
        # Extract speaker embeddings from reference speech
        speaker_embedding = None
        # Compute query
        query = dummy_prosody_embedding.reshape(1, 1, -1)
    keys = torch.tanh(token_embeddings).expand(1, -1, -1)
    gst_embedding_src = torch.vstack([
        tts_model.gst_layer.style_token_layer.attention(query, keys[:, i].unsqueeze(0))
        for i in range(tts_model.gst_style_tokens)
    ]).squeeze()
    # Extract GST embedding using reference style speech
    gst_embedding_tgt = tts_model.gst_layer(style_mel_spectrogram, speaker_embedding).squeeze()

    return gst_embedding_src, gst_embedding_tgt


@torch.no_grad()
def get_gst(
        reference_audio_path: str,
        tts_model: Tacotron2,
        tts_configs: AttrDict,
        tts_ap: Optional[AudioProcessor] = None,
        device: Optional[torch.device] = None
) -> Dict[str, float]:  # NOTE the GST encoder exists only as part of the Tacotron-2 TTS
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    # Compute GST sep. embeddings (unmixed and without reference prosody) and GST target embeddings (from reference prosody)
    gst_embedding_src, gst_embedding_tgt = get_gst_embedding_src_tgt(
        reference_audio_path, tts_model, tts_configs, tts_ap=tts_ap, device=device
    )
    # Use pseudo-inverse to derive the GST activations # NOTE this will result in an approximate solution
    gst: torch.tensor = torch.linalg.lstsq(gst_embedding_src.T, gst_embedding_tgt.unsqueeze(-1)).solution.squeeze()
    # Get GST on CPU if necessary
    if use_cuda:
        gst = gst.cpu()
    # Convert to dictionary
    gst = {str(i): val for i, val in enumerate(gst.tolist())}

    return gst


# Helper function for the Vocoder issue
def _interpolate_vocoder_input(scale_factor: List[float], spec: np.ndarray):
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    spec = torch.nn.functional.interpolate(spec, scale_factor=scale_factor, mode='bilinear').squeeze(0)

    return spec


@torch.no_grad()
def synthesise_speech(
        text: str,
        tts_model: Tacotron2,
        tts_configs: AttrDict,
        tts_ap: Optional[AudioProcessor] = None,
        vocoder_model: Optional[Union[FullbandMelganGenerator, Wavegrad, MultibandMelganGenerator]] = None,
        vocoder_configs: Optional[AttrDict] = None,
        vocoder_ap: Optional[AudioProcessor] = None,
        speaker_reference_clip_path: Optional[Union[List[str], str]] = None,
        speaker_embeddings: Optional[List[float]] = None,
        gst_reference_clip_path: Optional[str] = None,
        gst_style: Optional[Dict[str, float]] = None,  # e.g., {"0": .1, "1": 0, "2": -.1, "3": 0, "4": .1}
        denormalize: bool = True,
        device: Optional[torch.device] = None,
        out_path: Optional[str] = None,
        plot: bool = False,
        plot_path: Optional[str] = None,
        return_cahe: bool = False
) -> np.ndarray:  # NOTE there are some issues to check to ensure other options a part from the basic ones work
    # Input consistency check
    assert speaker_reference_clip_path is None or speaker_embeddings is None
    assert gst_reference_clip_path is None or gst_style is None
    # Prepare inputs for actual synthesis
    use_cuda = device.type == 'cuda'
    use_gl = vocoder_model is None
    if speaker_reference_clip_path is not None:
        speaker_embeddings = get_speaker_embedding(speaker_reference_clip_path)
    if gst_reference_clip_path is not None:
        gst_style = gst_reference_clip_path
    # Audio synthesis step
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(
        tts_model,
        text,
        tts_configs,
        use_cuda,
        tts_ap,
        style_wav=gst_style,
        use_griffin_lim=use_gl,
        speaker_embedding=speaker_embeddings
    )
    # Postprocessing
    if not use_gl:
        if denormalize:
            mel_postnet_spec = tts_ap._denormalize(mel_postnet_spec.T).T
            vocoder_input = vocoder_ap._normalize(mel_postnet_spec.T)
        else:
            vocoder_input = vocoder_ap._normalize(mel_postnet_spec.T)

        output_scale_factor = vocoder_configs.audio['sample_rate'] / tts_configs.audio['sample_rate']
        if output_scale_factor != 1.:
            scale_factor = [1., output_scale_factor]
            vocoder_input = _interpolate_vocoder_input(scale_factor, vocoder_input)
        else:
            vocoder_input = torch.FloatTensor(vocoder_input).unsqueeze(0)
        waveform = vocoder_model.inference(vocoder_input)
    # Get waveform on CPU if necessary
    if use_cuda and not use_gl:
        waveform = waveform.cpu()
    # Get waveform as a NumPy array if necessary (depends on Griffin-Limm use)
    if not use_gl:
        waveform = waveform.numpy()
    # Convert NumPy array into a flat array
    waveform = waveform.squeeze()

    # Save waveform to file if path is provided
    if out_path is not None:
        sr = vocoder_configs.audio['sample_rate'] if vocoder_configs is not None else tts_configs.audio['sample_rate']
        soundfile.write(out_path, waveform, sr)

    # Plot generated Mel Spectrogram
    if plot and mel_postnet_spec is not None:
        mel_spec = tts_ap._denormalize(mel_spec.T).T
        visualize(
            alignment,
            mel_postnet_spec,
            text,
            tts_ap.hop_length,
            tts_configs,
            stop_tokens,
            mel_spec,
            figsize=(32, 16),
            output_path=plot_path
        )
    if return_cahe:
        return waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs
    else:
        return waveform
