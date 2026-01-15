"""Test equivalence of einops.rearrange operations with reshape/permute.

This module verifies that einops.rearrange can be replaced with standard
PyTorch operations (view, reshape, permute) without any numerical differences.
"""

import torch
from einops import rearrange


def test_qkv_splitting_equivalence():
    """Test equivalence of QKV projection splitting operation.
    
    Original: rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=num_heads)
    Replacement: view + unbind + stack + permute
    """
    # Test parameters
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    d = embed_dim // num_heads
    
    # Create test input: [B, T, 3*H*D]
    projected = torch.randn(batch_size, seq_len, 3 * num_heads * d)
    
    # Original einops operation
    qkv_einops = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=num_heads)
    
    # Replacement using standard PyTorch operations
    # Step 1: Reshape to [B, T, 3, H, D]
    packed = projected.view(batch_size, seq_len, 3, num_heads, d)
    # Step 2: Unbind to get separate Q, K, V tensors
    q, k, v = torch.unbind(packed, dim=2)  # Each: [B, T, H, D]
    # Step 3: Stack and permute to match einops output shape [3, B, H, T, D]
    qkv_manual = torch.stack([q, k, v], dim=0)  # [3, B, T, H, D]
    qkv_manual = qkv_manual.permute(0, 1, 3, 2, 4)  # [3, B, H, T, D]
    
    # Verify equivalence
    assert qkv_einops.shape == qkv_manual.shape, (
        f"Shape mismatch: einops {qkv_einops.shape} vs manual {qkv_manual.shape}"
    )
    assert torch.allclose(qkv_einops, qkv_manual, atol=1e-6), (
        f"Numerical difference: max={torch.abs(qkv_einops - qkv_manual).max().item():.2e}"
    )
    
    print("✓ QKV splitting equivalence test passed")


def test_attention_output_reshaping_equivalence():
    """Test equivalence of attention output reshaping operation.
    
    Original: rearrange(x, "b h t d -> b t (h d)")
    Replacement: permute + reshape
    """
    # Test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64
    
    # Create test input: [B, H, T, D]
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Original einops operation
    out_einops = rearrange(x, "b h t d -> b t (h d)")
    
    # Replacement using standard PyTorch operations
    # Step 1: Permute to [B, T, H, D]
    x_permuted = x.permute(0, 2, 1, 3)
    # Step 2: Reshape to [B, T, H*D]
    out_manual = x_permuted.reshape(batch_size, seq_len, num_heads * head_dim)
    
    # Verify equivalence
    assert out_einops.shape == out_manual.shape, (
        f"Shape mismatch: einops {out_einops.shape} vs manual {out_manual.shape}"
    )
    assert torch.allclose(out_einops, out_manual, atol=1e-6), (
        f"Numerical difference: max={torch.abs(out_einops - out_manual).max().item():.2e}"
    )
    
    print("✓ Attention output reshaping equivalence test passed")


def test_all_equivalence():
    """Run all equivalence tests."""
    print("Running einops equivalence tests...")
    test_qkv_splitting_equivalence()
    test_attention_output_reshaping_equivalence()
    print("All equivalence tests passed! ✓")


if __name__ == "__main__":
    test_all_equivalence()
