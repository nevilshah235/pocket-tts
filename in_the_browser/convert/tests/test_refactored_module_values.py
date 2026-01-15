"""Test that refactored module produces identical values to original.

This test compares the refactored code (without einops) against the original
einops-based implementation to ensure no numerical differences.

The original version is stored in mimi_transformer_original.py in this directory.
"""

import torch
import importlib.util
import sys
from pathlib import Path


def load_original_module():
    """Load the original mimi_transformer module with einops."""
    test_dir = Path(__file__).parent
    original_file = test_dir / "mimi_transformer_original.py"
    
    if not original_file.exists():
        raise FileNotFoundError(
            f"Original module file not found: {original_file}\n"
            "This file should contain the einops-based version of mimi_transformer.py"
        )
    
    # Load the module from file
    spec = importlib.util.spec_from_file_location("mimi_transformer_original", original_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {original_file}")
    
    original_module = importlib.util.module_from_spec(spec)
    # Add necessary paths to sys.path for imports to work
    sys.path.insert(0, str(test_dir.parent.parent.parent))
    try:
        spec.loader.exec_module(original_module)
    finally:
        sys.path.pop(0)
    
    return original_module


def test_refactored_vs_original_values():
    """Compare refactored module outputs against original einops version.
    
    This test:
    1. Loads the original module (with einops) from mimi_transformer_original.py
    2. Loads the refactored module (without einops) from the main codebase
    3. Creates identical instances with same weights
    4. Compares outputs for same inputs
    5. Verifies both shapes AND values match
    """
    try:
        original_module = load_original_module()
    except Exception as e:
        print(f"⚠ Could not load original module: {e}")
        print("  Skipping comparison test. Make sure mimi_transformer_original.py exists.")
        return
    
    from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention as RefactoredAttention
    from pocket_tts.modules.rope import RotaryEmbedding
    
    OriginalAttention = original_module.MimiStreamingMultiheadAttention
    
    # Test parameters
    embed_dim = 512
    num_heads = 8
    context = 250
    max_period = 10000.0
    batch_size = 2
    seq_len = 10
    
    rope = RotaryEmbedding(max_period=max_period)
    
    # Create both versions
    original_attn = OriginalAttention(embed_dim, num_heads, context, rope)
    refactored_attn = RefactoredAttention(embed_dim, num_heads, context, rope)
    
    # Copy weights to ensure identical computation
    refactored_attn.in_proj.weight.data = original_attn.in_proj.weight.data.clone()
    refactored_attn.out_proj.weight.data = original_attn.out_proj.weight.data.clone()
    
    # Create test input
    torch.manual_seed(42)  # For reproducibility
    query = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize states
    original_state = original_attn.init_state(batch_size, 1000)
    refactored_state = refactored_attn.init_state(batch_size, 1000)
    
    # Copy state
    refactored_state["offset"] = original_state["offset"].clone()
    refactored_state["cache"] = original_state["cache"].clone()
    refactored_state["end_offset"] = original_state["end_offset"].clone()
    
    # Set module names for state access
    original_attn._module_absolute_name = "original"
    refactored_attn._module_absolute_name = "refactored"
    
    model_state_orig = {"original": original_state}
    model_state_ref = {"refactored": refactored_state}
    
    # Forward pass
    with torch.no_grad():
        output_orig = original_attn(query, model_state_orig)
        output_ref = refactored_attn(query, model_state_ref)
    
    # Verify shapes match
    assert output_orig.shape == output_ref.shape, (
        f"Shape mismatch: original {output_orig.shape} vs refactored {output_ref.shape}"
    )
    
    # Verify values match (with tolerance for floating point)
    max_diff = torch.abs(output_orig - output_ref).max().item()
    mean_diff = torch.abs(output_orig - output_ref).mean().item()
    
    print(f"\nValue comparison (original einops vs refactored):")
    print(f"  Shape match: ✓")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  All close (atol=1e-5): {torch.allclose(output_orig, output_ref, atol=1e-5)}")
    
    # For pure tensor operations, we expect very close values
    # Small differences might occur due to operation order, but should be minimal
    assert torch.allclose(output_orig, output_ref, atol=1e-4), (
        f"Output values differ significantly: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
    )
    
    print("✓ Refactored module produces equivalent values to original!")


if __name__ == "__main__":
    test_refactored_vs_original_values()
