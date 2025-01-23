import math
from queue import Queue
import numpy as np
import torch

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer

class ParlerTTSStreamer(BaseStreamer):
    def __init__(self):
        self.device = "cuda:0"
        torch_dtype = torch.float16
       
        repo_id = "ai4bharat/indic-parler-tts"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.description_tokenizer = AutoTokenizer.from_pretrained(repo_id)

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to_empty(device=self.device)
        self.decoder = self.model.decoder
        self.audio_encoder = self.model.audio_encoder
        self.generation_config = self.model.generation_config

        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        frame_rate = self.model.audio_encoder.config.frame_rate

        play_steps_in_s = 2.0
        play_steps = int(frame_rate * play_steps_in_s)

        self.play_steps = play_steps
        hop_length = math.floor(self.audio_encoder.config.sampling_rate / self.audio_encoder.config.frame_rate)
        self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0

        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = None

    def apply_delay_pattern_mask(self, input_ids):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device=self.device)
            
        _, delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            bos_token_id=self.generation_config.bos_token_id,
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        mask = (delay_pattern_mask != self.generation_config.bos_token_id) & (delay_pattern_mask != self.generation_config.pad_token_id)
        input_ids = input_ids[mask].reshape(1, self.decoder.num_codebooks, -1)
        input_ids = input_ids[None, ...].to(self.audio_encoder.device)

        decode_sequentially = (
            self.generation_config.bos_token_id in input_ids
            or self.generation_config.pad_token_id in input_ids
            or self.generation_config.eos_token_id in input_ids
        )
        
        if not decode_sequentially:
            output_values = self.audio_encoder.decode(input_ids, audio_scales=[None])
        else:
            sample = input_ids[:, 0]
            sample_mask = (sample >= self.audio_encoder.config.codebook_size).sum(dim=(0, 1)) == 0
            sample = sample[:, :, sample_mask]
            output_values = self.audio_encoder.decode(sample[None, ...], [None])

        audio_values = output_values.audio_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, new_tokens: torch.Tensor):
        # 1. Accumulate incoming tokens
        if self.token_cache is None:
            self.token_cache = new_tokens
        else:
            self.token_cache = torch.cat([self.token_cache, new_tokens], dim=-1)
    
        # 2. Check if we reached chunk size
        if self.token_cache.shape[-1] >= self.play_steps:
            # Decode everything
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
    
            # Slice out new portion
            new_audio = audio_values[self.already_emitted_samples : ]
    
            self.on_finalized_audio(new_audio)
            self.already_emitted_samples = audio_values.shape[0]
    
    def end(self):
        # Final decode
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            new_audio = audio_values[self.already_emitted_samples : ]
            self.on_finalized_audio(new_audio, stream_end=True)
        else:
            self.on_finalized_audio(np.array([]), stream_end=True)

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
