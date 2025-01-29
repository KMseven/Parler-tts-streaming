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
        
        # Updated to use v1 model
        repo_id = "ai4bharat/indic-parler-tts"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        # Initialize model
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo_id, 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False
        ).to(self.device)
        
        # Setup components and configurations
        self.decoder = self.model.decoder
        self.audio_encoder = self.model.audio_encoder
        self.generation_config = self.model.generation_config
        
        # Configure sampling and frame rates
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        frame_rate = self.model.audio_encoder.config.frame_rate
        
        # Setup streaming parameters
        play_steps_in_s = 2.0
        play_steps = int(frame_rate * play_steps_in_s)
        self.play_steps = play_steps
        
        # Configure stride
        hop_length = math.floor(self.audio_encoder.config.sampling_rate / self.audio_encoder.config.frame_rate)
        self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        
        # Initialize streaming variables
        self.token_cache = None
        self.to_yield = 0
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = None

    def apply_delay_pattern_mask(self, input_ids):
        # Build delay pattern mask
        _, delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            bos_token_id=self.generation_config.bos_token_id,
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        
        # Apply pattern mask
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        
        # Filter mask
        mask = (delay_pattern_mask != self.generation_config.bos_token_id) & (delay_pattern_mask != self.generation_config.pad_token_id)
        input_ids = input_ids[mask].reshape(1, self.decoder.num_codebooks, -1)
        input_ids = input_ids[None, ...]
        
        input_ids = input_ids.to(self.audio_encoder.device)
        
        # Decode based on token presence
        decode_sequentially = (
            self.generation_config.bos_token_id in input_ids
            or self.generation_config.pad_token_id in input_ids
            or self.generation_config.eos_token_id in input_ids
        )
        
        if not decode_sequentially:
            output_values = self.audio_encoder.decode(
                audio_codes=input_ids
            )
        else:
            sample = input_ids[:, 0]
            sample_mask = (sample >= self.audio_encoder.config.codebook_size).sum(dim=(0, 1)) == 0
            sample = sample[:, :, sample_mask]
            output_values = self.audio_encoder.decode(sample[None, ...])
        
        audio_values = output_values.audio_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
     
        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
            self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
        else:
            audio_values = np.zeros(self.to_yield)

        self.on_finalized_audio(audio_values[self.to_yield :], stream_end=True)
        
    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        else:
            return value
