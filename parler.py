import math
from queue import Queue
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from parler_tts import ParlerTTSForConditionalGeneration

class ParlerTTSStreamer(BaseStreamer):
    def __init__(self):
        self.device = "cuda:0"
        torch_dtype = torch.float16
        
        repo_id = "ai4bharat/indic-parler-tts"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo_id, 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False
        ).to(self.device)
        
        self.decoder = self.model.decoder
        self.audio_encoder = self.model.audio_encoder
        self.generation_config = self.model.generation_config
        
        # Audio configuration
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        self.frame_rate = self.model.audio_encoder.config.frame_rate
        
        # Each code is 0.011 seconds, so for 1 second we need ~91 tokens
        # We'll use a slightly smaller chunk for safety
        self.tokens_per_second = int(1 / 0.011)  # ≈91 tokens
        self.chunk_size = 86  # Tokens per chunk (approx 1 second of audio)
        
        # Initialize streaming variables
        self.token_cache = []
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = None

    def decode_chunk(self, tokens):
        """Decode a chunk of tokens into audio."""
        try:
            # Reshape tokens to match model's expected format
            tokens = tokens.reshape(1, -1, self.decoder.num_codebooks)
            tokens = tokens.to(self.device)
            
            # Decode to audio using the DAC model
            with torch.no_grad():
                audio = self.audio_encoder.decode(tokens).audio_values
            
            return audio[0, 0].cpu().float().numpy()
        except Exception as e:
            print(f"Error decoding chunk: {e}")
            print(f"Token shape: {tokens.shape}")
            raise e

    def put(self, token_ids):
        """Receive tokens from the model's generation process."""
        # Add new tokens to cache
        self.token_cache.append(token_ids.cpu())
        
        # If we have enough tokens for a chunk, process them
        if len(self.token_cache) >= self.chunk_size:
            # Convert list of tensors to single tensor
            tokens = torch.stack(self.token_cache)
            
            # Decode the chunk to audio
            audio = self.decode_chunk(tokens)
            
            # Queue the audio chunk
            self.on_finalized_audio(audio)
            
            # Clear the cache
            self.token_cache = []

    def end(self):
        """Handle any remaining tokens when generation ends."""
        if self.token_cache:
            # Process any remaining tokens
            tokens = torch.stack(self.token_cache)
            audio = self.decode_chunk(tokens)
            self.on_finalized_audio(audio)
            
        # Signal the end of streaming
        self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        """Queue the audio chunk for streaming."""
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next chunk of audio for streaming."""
        value = self.audio_queue.get(timeout=self.timeout)
        if value is self.stop_signal:
            raise StopIteration()
        return value
