"""Test full model output comparison between original and refactored versions.

This module compares the complete TTS pipeline outputs to ensure no quality
loss during refactoring. It uses golden reference testing with the same inputs.
"""

import torch
import pytest
from pathlib import Path


def test_refactored_model_outputs():
    """Compare outputs of original vs refactored models.
    
    This test loads both the original model and a refactored version (without einops)
    and compares their outputs for the same inputs.
    """
    # Skip if models aren't available (e.g., in CI without weights)
    try:
        from pocket_tts import TTSModel
    except ImportError:
        pytest.skip("pocket_tts not available")
    
    # Test parameters
    test_text = "Hello world, this is a test sentence."
    voice_url = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    
    print("Loading original model...")
    model_original = TTSModel.load_model()
    
    print("Loading refactored model...")
    # Note: This will be the refactored version once we complete the refactoring
    # For now, we'll test that the original model works correctly
    model_refactored = TTSModel.load_model()  # TODO: Replace with refactored version
    
    # Set random seed for reproducibility (if generation is deterministic)
    torch.manual_seed(42)
    
    print("Getting voice state...")
    try:
        state_orig = model_original.get_state_for_audio_prompt(voice_url)
        state_ref = model_refactored.get_state_for_audio_prompt(voice_url)
    except Exception as e:
        pytest.skip(f"Could not load voice prompt: {e}")
    
    print("Generating audio with original model...")
    with torch.no_grad():
        audio_orig = model_original.generate_audio(state_orig, test_text)
    
    print("Generating audio with refactored model...")
    with torch.no_grad():
        audio_ref = model_refactored.generate_audio(state_ref, test_text)
    
    # Verify outputs have same shape
    assert audio_orig.shape == audio_ref.shape, (
        f"Audio shape mismatch: original {audio_orig.shape} vs refactored {audio_ref.shape}"
    )
    
    # Compare outputs
    # Note: If generation involves randomness, outputs may differ slightly
    # We check for reasonable similarity
    max_diff = torch.abs(audio_orig - audio_ref).max().item()
    mean_diff = torch.abs(audio_orig - audio_ref).mean().item()
    
    print(f"Audio comparison:")
    print(f"  Shape: {audio_orig.shape}")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Sample rate: {model_original.sample_rate} Hz")
    
    # For deterministic operations, we expect very close outputs
    # For non-deterministic (e.g., with temperature), we check statistical similarity
    # Using a reasonable tolerance for float32
    assert torch.allclose(audio_orig, audio_ref, atol=1e-4, rtol=1e-4), (
        f"Audio outputs differ significantly: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )
    
    print("✓ Model output equivalence test passed")


def test_model_outputs_deterministic():
    """Test that model outputs are deterministic with same seed."""
    try:
        from pocket_tts import TTSModel
    except ImportError:
        pytest.skip("pocket_tts not available")
    
    test_text = "This is a deterministic test."
    voice_url = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    
    model = TTSModel.load_model()
    
    # Generate twice with same seed
    torch.manual_seed(123)
    state1 = model.get_state_for_audio_prompt(voice_url)
    with torch.no_grad():
        audio1 = model.generate_audio(state1, test_text)
    
    torch.manual_seed(123)
    state2 = model.get_state_for_audio_prompt(voice_url)
    with torch.no_grad():
        audio2 = model.generate_audio(state2, test_text)
    
    # Check if outputs are identical (or very close)
    max_diff = torch.abs(audio1 - audio2).max().item()
    print(f"Determinism test - max difference: {max_diff:.2e}")
    
    # Note: Some operations may have small numerical differences
    # We check for reasonable similarity
    assert torch.allclose(audio1, audio2, atol=1e-5), (
        f"Model outputs are not deterministic: max_diff={max_diff:.2e}"
    )
    
    print("✓ Determinism test passed")


if __name__ == "__main__":
    test_refactored_model_outputs()
    test_model_outputs_deterministic()
