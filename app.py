import io
import base64
import numpy as np
from threading import Thread
from pydub import AudioSegment
from parler import ParlerTTSStreamer

class InferlessPythonModel:
    def initialize(self):
        # Initialize the ParlerTTSStreamer object
        self.streamer = ParlerTTSStreamer()

    def numpy_to_mp3(self, audio_array, sampling_rate):
        if len(audio_array) == 0:
            return b""
        # Convert numpy array to MP3 format
        if np.issubdtype(audio_array.dtype, np.floating):
            # Normalize floating-point audio data to 16-bit integer range
            max_val = np.max(np.abs(audio_array))
            audio_array = (audio_array / max_val) * 32767
            audio_array = audio_array.astype(np.int16)
        
        # Create an AudioSegment object from the numpy array
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sampling_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )
        
        # Export the AudioSegment to MP3 format
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="320k")
        mp3_bytes = mp3_io.getvalue()
        mp3_io.close()
        
        return mp3_bytes

    
    def infer(self, inputs, stream_output_handler):
        self.streamer.token_cache = None
        self.streamer.to_yield = 0
        
        input_value = inputs["input_value"]  # This should be the description
        prompt_value = inputs["prompt_value"]  # This is the text to speak
        
        # Use separate tokenizers for description and prompt
        description_inputs = self.streamer.description_tokenizer(input_value, return_tensors="pt").to(self.streamer.device)
        prompt_inputs = self.streamer.tokenizer(prompt_value, return_tensors="pt").to(self.streamer.device)
        
        generation_kwargs = dict(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
            streamer=self.streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10)
        
        thread = Thread(target=self.streamer.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_audio in self.streamer:
            mp3_bytes = self.numpy_to_mp3(new_audio, sampling_rate=self.streamer.sampling_rate)
            mp3_str = base64.b64encode(mp3_bytes).decode('utf-8')
            
            output_dict = {"OUT": mp3_str}
            stream_output_handler.send_streamed_output(output_dict)
        
        thread.join()
        stream_output_handler.finalise_streamed_output()

    
    def finalize(self):
        # Clean up resources
        self.streamer = None
