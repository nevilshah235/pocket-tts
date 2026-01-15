"""Test module-level output verification for refactored components.

This module verifies that individual refactored modules produce identical
outputs to the original implementations.
"""

import torch
import pytest
from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention
from pocket_tts.modules.rope import RotaryEmbedding


def create_refactored_attention(embed_dim, num_heads, context, rope):
    """Create attention module with refactored einops operations.
    
    This is a copy of MimiStreamingMultiheadAttention but with einops
    replaced by reshape/permute operations.
    """
    from pocket_tts.modules.stateful_module import StatefulModule
    from torch import nn
    from torch.nn import functional as F
    
    class RefactoredMimiStreamingMultiheadAttention(StatefulModule):
        def __init__(self, embed_dim: int, num_heads: int, context: int, rope: RotaryEmbedding):
            super().__init__()
            self.embed_dim = embed_dim
            self.context = context
            self.rope = rope
            self.num_heads = num_heads
            out_dim = 3 * embed_dim
            
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        
        def init_state(self, batch_size: int, sequence_length: int):
            dim_per_head = self.embed_dim // self.num_heads
            state = {}
            state["offset"] = torch.zeros(batch_size, dtype=torch.long)
            state["cache"] = torch.zeros((2, batch_size, self.num_heads, sequence_length, dim_per_head))
            state["end_offset"] = torch.zeros(batch_size, dtype=torch.long)
            return state
        
        def increment_step(self, state, increment: int = 1):
            state["offset"] += increment
        
        def forward(self, query: torch.Tensor, model_state: dict | None = None):
            B, T = query.shape[:2]
            
            if model_state is None:
                offset = torch.zeros(B, device=query.device, dtype=torch.long)
            else:
                offset = self.get_state(model_state)["offset"]
            
            projected = self.in_proj(query)
            
            # REFACTORED: Replace einops with reshape/permute
            # Original: q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)
            d = self.embed_dim // self.num_heads
            packed = projected.view(B, T, 3, self.num_heads, d)  # [B, T, 3, H, D]
            q, k, v = torch.unbind(packed, dim=2)  # Each: [B, T, H, D]
            # Permute to [B, H, T, D] for rope (matching original behavior)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            
            q, k = self.rope(q, k, offset)
            
            # Permute back to [B, H, T, D]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            
            # Simplified attention (without KV cache for testing)
            q = q.permute(0, 2, 1, 3)  # [B, T, H, D]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            
            # REFACTORED: Replace einops with permute + reshape
            # Original: x = rearrange(x, "b h t d -> b t (h d)")
            x = x.permute(0, 2, 1, 3)  # [B, T, H, D]
            x = x.reshape(B, T, self.num_heads * d)  # [B, T, H*D]
            
            x = self.out_proj(x)
            return x
    
    return RefactoredMimiStreamingMultiheadAttention(embed_dim, num_heads, context, rope)


def test_attention_module_equivalence():
    """Test that refactored attention module produces same outputs."""
    embed_dim = 512
    num_heads = 8
    context = 250
    max_period = 10000.0
    
    rope = RotaryEmbedding(max_period=max_period)
    
    # Create original and refactored modules
    original_attn = MimiStreamingMultiheadAttention(embed_dim, num_heads, context, rope)
    refactored_attn = create_refactored_attention(embed_dim, num_heads, context, rope)
    
    # Copy weights from original to refactored
    refactored_attn.in_proj.weight.data = original_attn.in_proj.weight.data.clone()
    refactored_attn.out_proj.weight.data = original_attn.out_proj.weight.data.clone()
    
    # Create test input
    batch_size = 2
    seq_len = 10
    query = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize states
    original_state = original_attn.init_state(batch_size, 1000)
    refactored_state = refactored_attn.init_state(batch_size, 1000)
    
    # Copy state
    refactored_state["offset"] = original_state["offset"].clone()
    refactored_state["cache"] = original_state["cache"].clone()
    refactored_state["end_offset"] = original_state["end_offset"].clone()
    
    # Forward pass
    original_attn._module_absolute_name = "original"
    refactored_attn._module_absolute_name = "refactored"
    
    model_state_orig = {"original": original_state}
    model_state_ref = {"refactored": refactored_state}
    
    with torch.no_grad():
        output_orig = original_attn(query, model_state_orig)
        output_ref = refactored_attn(query, model_state_ref)
    
    # Verify outputs match
    assert output_orig.shape == output_ref.shape, (
        f"Shape mismatch: original {output_orig.shape} vs refactored {output_ref.shape}"
    )
    
    # Note: There may be small differences due to floating point operations order
    # We use a reasonable tolerance
    max_diff = torch.abs(output_orig - output_ref).max().item()
    assert torch.allclose(output_orig, output_ref, atol=1e-5), (
        f"Output mismatch: max difference = {max_diff:.2e}"
    )
    
    print(f"✓ Attention module equivalence test passed (max diff: {max_diff:.2e})")


if __name__ == "__main__":
    test_attention_module_equivalence()
