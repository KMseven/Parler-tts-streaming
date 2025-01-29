import io
import base64
import numpy as np
from threading import Thread
from pydub import AudioSegment
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
import torch

class InferlessPythonModel:
    def initialize(self):
        self.device = "cuda:0"
        torch_dtype = torch.float16
        repo_id = "ai4bharat/indic-parler-tts"
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False
        ).to(self.device)
        
        # Get audio configuration
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        self.frame_rate = self.model.audio_encoder.config.frame_rate

    def numpy_to_mp3(self, audio_array, sampling_rate):
        if audio_array.size == 0:
            return b''
            
        # Normalize and convert to int16
        max_val = np.max(np.abs(audio_array)) if audio_array.size > 0 else 1
        audio_array = (audio_array / max_val * 32767).astype(np.int16)
        
        # Create AudioSegment and export to MP3
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sampling_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )
        
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="320k")
        return mp3_io.getvalue()

    def infer(self, inputs, stream_output_handler):
        # Extract inputs
        input_value = inputs["input_value"]
        prompt_value = inputs["prompt_value"]
        chunk_size_in_s = 0.5  # Can be adjusted based on needs
        
        # Create streamer
        play_steps = int(self.frame_rate * chunk_size_in_s)
        streamer = ParlerTTSStreamer(
            self.model, 
            device=self.device,
            play_steps=play_steps
        )
        
        # Prepare inputs
        inputs_ = self.tokenizer(input_value, return_tensors="pt").to(self.device)
        prompt = self.tokenizer(prompt_value, return_tensors="pt").to(self.device)
        
        # Setup generation kwargs
        generation_kwargs = dict(
            input_ids=inputs_.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs_.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10
        )
        
        # Start generation thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Process audio chunks
        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break
                
            # Convert to MP3 and encode as base64
            mp3_bytes = self.numpy_to_mp3(new_audio, self.sampling_rate)
            mp3_str = base64.b64encode(mp3_bytes).decode('utf-8')
            
            # Send chunk
            output_dict = {"OUT": mp3_str}
            stream_output_handler.send_streamed_output(output_dict)
        
        # Wait for generation to complete
        thread.join()
        
        # Finalize streaming
        stream_output_handler.finalise_streamed_output()

    def finalize(self):
        self.model = None
        self.tokenizer = None
