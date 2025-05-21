import os
import torch
import torchaudio
import soundfile as sf
import io
from fastapi import WebSocket
from transformers import AutoProcessor, SeamlessM4Tv2Model
from utils import send_audio_to_client
from config import SAVE_CHUNKS
from wav2lip_inferencer import Wav2LipInferencer

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Translator:
    def __init__(self, model_name="facebook/seamless-m4t-v2-large", target_lang="hin"):
        """
        Initialize the translator with the specified model and target language.
        
        Args:
            model_name: Model name/path to load from HuggingFace
            target_lang: Target language code for translation
        """
        self.model_name = model_name
        self.target_lang = target_lang
        self.processor = None
        self.model = None
        self.output_dir = "converted_chunks"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Wav2LipInferencer once
        self.wav2lip_inferencer = Wav2LipInferencer(
            checkpoint_path='checkpoints/wav2lip_gan.pth'
        )
        self.REFERENCE_VIDEO_PATH = 'test_input/demo8.mp4'
        self.RESULTS_DIR = 'results'
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
    def load_model(self):
        """Load the translation model and processor"""
        print(f"Loading {self.model_name} model...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SeamlessM4Tv2Model.from_pretrained(self.model_name).to(device)
        print("Model loaded successfully")
        
    async def translate_audio(self, original_filepath: str, output_filename: str, websocket: WebSocket = None):
        """
        Translate audio from the input file to the target language.
        
        Args:
            original_filepath: Path to the original audio file
            output_filename: Name to use for the translated output file
            websocket: WebSocket connection to send the translated audio to
            
        Returns:
            The translated audio as a numpy array
        """
        # Make sure model is loaded
        if self.processor is None or self.model is None:
            self.load_model()
            
        # Load and preprocess audio
        audio, orig_freq = torchaudio.load(original_filepath)
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)

        # Ensure mono channel
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Process through model
        audio_inputs = self.processor(audios=audio, return_tensors="pt").to(device)
        translated_audio = self.model.generate(
            **audio_inputs, tgt_lang=self.target_lang
        )[0].cpu().numpy().squeeze()

        # Generate output path
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Get sample rate from model config
        sample_rate = self.model.config.sampling_rate
        
        if SAVE_CHUNKS:
            # Save translated audio to disk if SAVE_CHUNKS is True
            sf.write(output_path, translated_audio, sample_rate)
            print(f"Translated and saved: {output_path}")
            
            # Run lipsync on the translated chunk
            lipsynced_output = os.path.join(
                self.RESULTS_DIR, f"lipsynced_{output_filename.replace('.wav', '.mp4')}"
            )
            try:
                self.wav2lip_inferencer.infer(
                    face_path=self.REFERENCE_VIDEO_PATH,
                    audio_path=output_path,
                    outfile=lipsynced_output
                )
                print(f"Lipsynced video saved: {lipsynced_output}")
            except Exception as e:
                print(f"Lipsync failed for {output_path}: {e}")
            
            # Send to client if websocket provided
            if websocket:
                await send_audio_to_client(websocket, output_path)
        else:
            # Don't save to disk, but still send to client
            print(f"Translated audio (not saved to disk)")
            
            if websocket:
                # Create in-memory WAV file
                buffer = io.BytesIO()
                sf.write(buffer, translated_audio, sample_rate, format='WAV')
                buffer.seek(0)
                
                # Send directly from memory without saving to disk
                await send_audio_to_client(websocket, buffer=buffer)
            
        return translated_audio 