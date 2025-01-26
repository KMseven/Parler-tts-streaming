import math
from queue import Queue
import numpy as np
import torch
from transformers import AutoTokenizer, AutoFeatureExtractor
from transformers.generation.streamers import BaseStreamer
from parler_tts import ParlerTTSForConditionalGeneration

class ParlerTTSStreamer(BaseStreamer):
    def __init__(self):
        self.device = "cuda:0"
        torch_dtype = torch.float16
        
        repo_id = "parler-tts/parler-tts-mini-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
        
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo_id, 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.decoder = self.model.decoder
        self.audio_encoder = self.model.audio_encoder
        self.generation_config = self.model.generation_config
        self.sampling_rate = self.model.config.sampling_rate
        
        play_steps_in_s = 0.25  # Reduced from 2.0 for better responsiveness
        frame_rate = self.model.audio_encoder.config.frame_rate
        play_steps = int(frame_rate * play_steps_in_s)
        
        hop_length = math.floor(self.audio_encoder.config.sampling_rate / self.audio_encoder.config.frame_rate)
        self.play_steps = play_steps
        self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 3  # Changed from 6 to 3
        
        self.token_cache = None
        self.to_yield = 0
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = None

    def apply_delay_pattern_mask(self, input_ids):
        _, delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            bos_token_id=self.generation_config.bos_token_id,
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        mask = (delay_pattern_mask != self.generation_config.bos_token_id) & (delay_pattern_mask != self.generation_config.pad_token_id)
        input_ids = input_ids[mask].reshape(1, self.decoder.num_codebooks, -1)
        input_ids = input_ids[None, ...]
        input_ids = input_ids.to(self.audio_encoder.device)

        output_values = self.audio_encoder.decode(
            audio_codes=input_ids,
            audio_scales=[None],
        ).audio_values
        
        audio_values = output_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("ParlerTTSStreamer only supports batch size 1")
        
        value_expanded = value[:, None] if value.dim() == 1 else value
        if self.token_cache is None:
            self.token_cache = value_expanded
        else:
            self.token_cache = torch.concatenate([self.token_cache, value_expanded], dim=-1)
            
        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            if len(audio_values) > 0:  # Add check for empty audio
                self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
                self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            if len(audio_values) > 0:  # Add check for empty audio
                self.on_finalized_audio(audio_values[self.to_yield:], stream_end=True)
        self.audio_queue.put(self.stop_signal)

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        if audio is None or len(audio) == 0:
            return
            
        # Normalize audio to 16-bit PCM range
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        # Buffer the audio for smoother playback
        self.audio_buffer = np.concatenate((self.audio_buffer, audio)) if hasattr(self, 'audio_buffer') else audio
        
        # Only send to queue when buffer reaches sufficient size
        buffer_size = int(self.sampling_rate * 0.2)  # 200ms buffer
        while len(self.audio_buffer) >= buffer_size:
            chunk = self.audio_buffer[:buffer_size]
            self.audio_buffer = self.audio_buffer[buffer_size:]
            self.audio_queue.put(chunk, timeout=self.timeout)
            
        if stream_end and hasattr(self, 'audio_buffer') and len(self.audio_buffer) > 0:
            self.audio_queue.put(self.audio_buffer, timeout=self.timeout)
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        return value
