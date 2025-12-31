import gradio as gr
import numpy as np
import librosa
from audiosr import build_model
from audiosr.pipeline import super_resolution_from_waveform
import os

def calculate_amplitude_stats(audio):
    """Calculate amplitude statistics for audio normalization"""
    rms = np.sqrt(np.mean(np.square(audio)))
    peak = np.max(np.abs(audio))
    return rms, peak

def normalize_chunk_amplitude(processed_chunk, original_chunk):
    """Normalize processed chunk to match original chunk's amplitude characteristics"""
    orig_rms, orig_peak = calculate_amplitude_stats(original_chunk)
    proc_rms, proc_peak = calculate_amplitude_stats(processed_chunk)
    
    # Avoid division by zero
    if proc_rms < 1e-8:
        return processed_chunk
    
    # Calculate scaling factor based on RMS ratio
    scale_factor = orig_rms / proc_rms
    
    # Apply scaling while ensuring we don't exceed the original peak ratio
    peak_ratio = orig_peak / proc_peak if proc_peak > 0 else 1
    scale_factor = min(scale_factor, peak_ratio)
    
    return processed_chunk * scale_factor

def process_chunk_direct(audiosr, chunk, original_length, sr, guidance_scale, ddim_steps):
    """
    Process a single chunk directly without saving to file.
    
    Args:
        audiosr: The model
        chunk: numpy array (audio chunk)
        original_length: Original length of the chunk
        sr: Sample rate
        guidance_scale: Guidance scale
        ddim_steps: DDIM steps
        
    Returns:
        Processed chunk as numpy array
    """
    adjusted_ddim_steps = min(ddim_steps - 2, 998)
    
    # Process chunk directly (no file I/O)
    processed = super_resolution_from_waveform(
        audiosr,
        chunk,
        guidance_scale=guidance_scale,
        ddim_steps=adjusted_ddim_steps
    )
    
    # Normalize amplitude
    result = normalize_chunk_amplitude(processed, chunk)
    
    # Trim to original length
    result = np.squeeze(result)
    if len(result) > original_length:
        result = result[:original_length]
    
    return result


def process_audio_channel(audiosr, audio_channel, sr, guidance_scale, ddim_steps):
    """Process a single audio channel"""
    # Calculate chunk parameters
    chunk_duration = 5.1  # seconds
    chunk_size = int(chunk_duration * sr)
    overlap_duration = 0.5  # 500ms overlap
    overlap_size = int(overlap_duration * sr)
    
    # Calculate number of chunks
    total_samples = len(audio_channel)
    num_chunks = int(np.ceil(total_samples / (chunk_size - overlap_size)))
    
    print(f"Total chunks to process: {num_chunks}")
    
    # Process chunks sequentially
    processed_results = []
    
    for i in range(num_chunks):
        start = i * (chunk_size - overlap_size)
        end = min(start + chunk_size, total_samples)
        chunk = audio_channel[start:end]
        orig_len = len(chunk)
        
        print(f"\nChunk {i+1}/{num_chunks}: {start/sr:.2f}s - {end/sr:.2f}s ({(end-start)/sr:.2f}s)")
        
        # Use direct waveform processing (no file I/O)
        result = process_chunk_direct(
            audiosr, chunk, orig_len, sr, guidance_scale, ddim_steps
        )
        processed_results.append(result)
    
    # Apply crossfade and concatenate
    processed_chunks = []
    for i, result in enumerate(processed_results):
        # Ensure processed chunk is 2D
        processed_chunk = np.squeeze(result)
        if len(processed_chunk.shape) == 1:
            processed_chunk = processed_chunk.reshape(1, -1)
        
        # Apply crossfade for overlapping regions (except for first chunk)
        if i > 0:
            actual_overlap_size = overlap_size
            actual_overlap_size = min(actual_overlap_size, processed_chunk.shape[1], processed_chunks[-1].shape[1])
            
            # Create fade curves
            fade_in = np.linspace(0, 1, actual_overlap_size).reshape(1, -1)
            fade_out = np.linspace(1, 0, actual_overlap_size).reshape(1, -1)
            
            # Get overlapping regions
            current_overlap = processed_chunk[:, :actual_overlap_size]
            previous_overlap = processed_chunks[-1][:, -actual_overlap_size:]
            
            # Adjust fade curves based on RMS ratio
            current_rms = np.sqrt(np.mean(np.square(current_overlap)))
            previous_rms = np.sqrt(np.mean(np.square(previous_overlap)))
            
            if current_rms > 0 and previous_rms > 0:
                rms_ratio = np.sqrt(previous_rms / current_rms)
                fade_in = fade_in * rms_ratio
            
            # Apply crossfade
            processed_chunk[:, :actual_overlap_size] *= fade_in
            processed_chunks[-1][:, -actual_overlap_size:] *= fade_out
            processed_chunks[-1][:, -actual_overlap_size:] += processed_chunk[:, :actual_overlap_size]
            processed_chunk = processed_chunk[:, actual_overlap_size:]
        
        processed_chunks.append(processed_chunk)
    
    print(f"\nConcatenating {len(processed_chunks)} processed chunks...")
    return np.concatenate(processed_chunks, axis=1)

def normalize_audio(audio):
    """Normalize audio to be within [-1, 1] range"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def convert_audio_for_gradio(audio):
    """Convert audio to the format expected by Gradio"""
    # Ensure audio is in float32 format
    audio = audio.astype(np.float32)
    # Normalize to [-1, 1] range
    audio = normalize_audio(audio)
    # Transpose to (samples, channels) format if needed
    if audio.shape[0] == 2:  # If first dimension is channels
        audio = audio.T
    return audio

def inference(audio_file, model_name, guidance_scale, ddim_steps):
    # Initialize the model
    audiosr = build_model(model_name=model_name)
    
    # Load the audio file with original number of channels
    audio, sr = librosa.load(audio_file, sr=48000, mono=False)
    
    # Convert to stereo if mono
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])
    
    print(f"\nProcessing audio file of length: {audio.shape[1]/sr:.2f} seconds")
    print(f"Number of channels: {audio.shape[0]}")
    
    # Process each channel separately
    processed_channels = []
    for channel_idx in range(audio.shape[0]):
        print(f"\nProcessing channel {channel_idx + 1}")
        channel_audio = audio[channel_idx]
        processed_channel = process_audio_channel(
            audiosr, channel_audio, sr, guidance_scale, ddim_steps
        )
        # Ensure the channel is 1D
        processed_channel = processed_channel.squeeze()
        processed_channels.append(processed_channel)
    
    # Stack channels for stereo output (shape will be [2, samples])
    if len(processed_channels[0].shape) > 1:
        # If channels are 2D, take the first row
        processed_channels = [channel[0] if len(channel.shape) > 1 else channel for channel in processed_channels]
    
    final_audio = np.stack(processed_channels)
    
    # Convert audio to the format expected by Gradio
    final_audio = convert_audio_for_gradio(final_audio)
    
    print(f"Final audio shape: {final_audio.shape}")
    print(f"Final audio length: {final_audio.shape[0]/sr:.2f} seconds")
    print(f"Audio value range: [{final_audio.min():.3f}, {final_audio.max():.3f}]")
    print(f"Audio dtype: {final_audio.dtype}")
    
    return (48000, final_audio)

iface = gr.Interface(
    fn=inference, 
    inputs=[
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(["basic", "speech"], value="basic", label="Model"),
        gr.Slider(1, 10, value=2.6, step=0.1, label="Guidance Scale"),  
        gr.Slider(1, 100, value=100, step=1, label="DDIM Steps")
    ],
    outputs=gr.Audio(type="numpy", label="Output Audio"),
    title="AudioSR - Audio Super Resolution",
    description="音频超分辨率处理。直接处理波形数据，无需临时文件I/O。"
)

iface.launch()