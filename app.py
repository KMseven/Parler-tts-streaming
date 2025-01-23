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
       mp3_io = None
       try:
           if not isinstance(audio_array, np.ndarray):
               raise ValueError("Invalid audio array type")
               
           if audio_array.size == 0:
               raise ValueError("Empty audio array received")
               
           if np.issubdtype(audio_array.dtype, np.floating):
               max_val = np.max(np.abs(audio_array))
               if max_val == 0:
                   max_val = 1  # Prevent division by zero
               audio_array = (audio_array / max_val) * 32767
               audio_array = audio_array.astype(np.int16)
           
           audio_segment = AudioSegment(
               audio_array.tobytes(),
               frame_rate=sampling_rate,
               sample_width=audio_array.dtype.itemsize,
               channels=1
           )
           
           mp3_io = io.BytesIO()
           audio_segment.export(mp3_io, format="mp3", bitrate="320k")
           mp3_bytes = mp3_io.getvalue()
           return mp3_bytes
    
       except Exception as e:
           raise RuntimeError(f"Failed to convert audio to MP3: {str(e)}")
           
       finally:
           if mp3_io:
               mp3_io.close()

    
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
