import gradio as gr
import numpy as np
import librosa
from audiosr import super_resolution, super_resolution_batch, build_model
import tempfile
import soundfile as sf
import os
import torch

def detect_audio_end(audio, sr, window_size=2048, hop_length=512, threshold_db=-50):
    """Detect the end of actual audio content using RMS energy"""
    # Calculate RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=window_size, hop_length=hop_length)[0]
    
    # Convert to dB
    db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Find the last frame above threshold
    valid_frames = np.where(db > threshold_db)[0]
    
    if len(valid_frames) > 0:
        last_frame = valid_frames[-1]
        # Convert frame index to sample index
        last_sample = (last_frame + 1) * hop_length
        return last_sample
    return len(audio)

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

def process_chunks_batch(audiosr, chunks, original_lengths, sr, guidance_scale, ddim_steps):
    """
    Process multiple chunks in a single batch using GPU parallelization.
    
    Args:
        audiosr: The model
        chunks: List of numpy arrays (audio chunks)
        original_lengths: List of original lengths for each chunk
        sr: Sample rate
        guidance_scale: Guidance scale
        ddim_steps: DDIM steps
        
    Returns:
        List of processed chunks as numpy arrays
    """
    adjusted_ddim_steps = min(ddim_steps - 2, 998)
    
    # Process all chunks in one batch
    processed_chunks = super_resolution_batch(
        audiosr,
        chunks,
        guidance_scale=guidance_scale,
        ddim_steps=adjusted_ddim_steps
    )
    
    # Post-process each chunk
    results = []
    for i, (processed, original, orig_len) in enumerate(zip(processed_chunks, chunks, original_lengths)):
        # Normalize amplitude
        result = normalize_chunk_amplitude(processed, original)
        
        # Trim to original length
        result = np.squeeze(result)
        if len(result) > orig_len:
            result = result[:orig_len]
        
        results.append(result)
    
    return results


def process_chunk(audiosr, chunk, sr, guidance_scale, ddim_steps, is_last_chunk=False, target_length=None):
    """Process a single chunk (fallback for non-batch processing)"""
    # Create a temporary directory in the current working directory
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a unique temporary file path
    temp_path = os.path.join(temp_dir, f"chunk_{np.random.randint(0, 1000000)}.wav")
    
    # Store the original input chunk length
    original_chunk_length = len(chunk)
    
    try:
        # Save chunk to temporary file
        sf.write(temp_path, chunk, sr)
        
        # For the last chunk, adjust ddim_steps based on length
        if is_last_chunk:
            chunk_duration = len(chunk) / sr
            max_steps = min(ddim_steps - 2, 998)
            adjusted_ddim_steps = max(10, min(max_steps, int(ddim_steps * (chunk_duration / 5.1))))
            print(f"Adjusted ddim_steps for last chunk: {adjusted_ddim_steps}")
        else:
            adjusted_ddim_steps = min(ddim_steps - 2, 998)
        
        # Process the chunk
        processed_chunk = super_resolution(
            audiosr,
            temp_path,
            guidance_scale=guidance_scale,
            ddim_steps=adjusted_ddim_steps
        )
        
        result = processed_chunk
        
        # Normalize the processed chunk's amplitude relative to input chunk
        result = normalize_chunk_amplitude(result, chunk)
        
        # Trim to original length
        result = np.squeeze(result)
        if len(result) > original_chunk_length:
            result = result[:original_chunk_length]
            print(f"Trimmed chunk from {len(processed_chunk.squeeze())} to {original_chunk_length} samples to match input duration")
        
        # For the last chunk, optionally trim further
        if is_last_chunk and target_length is not None:
            target_output_length = target_length
            audio_end = detect_audio_end(result, sr)
            end_point = min(audio_end, target_output_length, len(result))
            result = result[:end_point]
            print(f"Adjusted last chunk length to {end_point} samples")
        
        return result
    
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_path}: {e}")

def process_audio_channel(audiosr, audio_channel, sr, guidance_scale, ddim_steps, batch_size=1):
    """Process a single audio channel with optional batch processing"""
    # Calculate chunk parameters
    chunk_duration = 5.1  # seconds
    chunk_size = int(chunk_duration * sr)
    overlap_duration = 0.5  # 500ms overlap
    overlap_size = int(overlap_duration * sr)
    
    # Calculate number of chunks
    total_samples = len(audio_channel)
    num_chunks = int(np.ceil(total_samples / (chunk_size - overlap_size)))
    
    print(f"Total chunks to process: {num_chunks}")
    print(f"Batch size: {batch_size}")
    
    # Collect all chunks first
    all_chunks = []
    all_original_lengths = []
    chunk_boundaries = []
    
    for i in range(num_chunks):
        start = i * (chunk_size - overlap_size)
        end = min(start + chunk_size, total_samples)
        chunk = audio_channel[start:end]
        all_chunks.append(chunk)
        all_original_lengths.append(len(chunk))
        chunk_boundaries.append((start, end))
    
    # Process chunks in batches
    processed_results = []
    num_batches = int(np.ceil(num_chunks / batch_size))
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_chunks)
        
        batch_chunks = all_chunks[batch_start:batch_end]
        batch_lengths = all_original_lengths[batch_start:batch_end]
        
        print(f"\n{'='*50}")
        print(f"Processing batch {batch_idx+1}/{num_batches} (chunks {batch_start+1}-{batch_end}/{num_chunks})")
        print(f"{'='*50}")
        
        if batch_size > 1 and len(batch_chunks) > 1:
            # Use batch processing
            batch_results = process_chunks_batch(
                audiosr, batch_chunks, batch_lengths, sr, guidance_scale, ddim_steps
            )
            processed_results.extend(batch_results)
        else:
            # Process single chunk
            for j, (chunk, orig_len) in enumerate(zip(batch_chunks, batch_lengths)):
                chunk_idx = batch_start + j
                start, end = chunk_boundaries[chunk_idx]
                is_last = (chunk_idx == num_chunks - 1)
                
                print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks}")
                print(f"Chunk time range: {start/sr:.2f}s to {end/sr:.2f}s of total {total_samples/sr:.2f}s")
                
                if is_last:
                    remaining_samples = total_samples - start
                    result = process_chunk(audiosr, chunk, sr, guidance_scale, ddim_steps,
                                         is_last_chunk=True, target_length=remaining_samples)
                else:
                    result = process_chunk(audiosr, chunk, sr, guidance_scale, ddim_steps,
                                         is_last_chunk=False)
                processed_results.append(result)
        
        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
        print(f"Chunk {i+1} processed and crossfaded successfully")
    
    print("\nConcatenating processed chunks...")
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

def inference(audio_file, model_name, guidance_scale, ddim_steps, batch_size):
    # Initialize the model
    audiosr = build_model(model_name=model_name)
    
    # Load the audio file with original number of channels
    audio, sr = librosa.load(audio_file, sr=48000, mono=False)
    
    # Convert to stereo if mono
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])
    
    print(f"\nProcessing audio file of length: {audio.shape[1]/sr:.2f} seconds")
    print(f"Number of channels: {audio.shape[0]}")
    print(f"Batch size: {batch_size}")
    
    # Process each channel separately
    processed_channels = []
    for channel_idx in range(audio.shape[0]):
        print(f"\nProcessing channel {channel_idx + 1}")
        channel_audio = audio[channel_idx]
        processed_channel = process_audio_channel(
            audiosr, channel_audio, sr, guidance_scale, ddim_steps, 
            batch_size=int(batch_size)
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
    
    # Clean up temporary directory
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    try:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")
    
    return (48000, final_audio)

iface = gr.Interface(
    fn=inference, 
    inputs=[
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(["basic", "speech"], value="basic", label="Model"),
        gr.Slider(1, 10, value=2.6, step=0.1, label="Guidance Scale"),  
        gr.Slider(1, 100, value=100, step=1, label="DDIM Steps"),
        gr.Slider(1, 8, value=1, step=1, label="Batch Size", 
                  info="同时处理多个音频块以加速处理。增大此值需要更多显存。RTX 5070 Ti (16GB) 建议 2-4")
    ],
    outputs=gr.Audio(type="numpy", label="Output Audio"),
    title="AudioSR - Audio Super Resolution",
    description="音频超分辨率处理。支持批量处理以加速长音频处理。"
)

# Create temp directory on startup
temp_dir = os.path.join(os.getcwd(), "temp_audio")
os.makedirs(temp_dir, exist_ok=True)

iface.launch()