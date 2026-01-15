"""Utilities for flattening and unflattening state dictionaries for ONNX export.

ONNX doesn't support Python dictionaries directly, so we need to convert nested
state dictionaries into flat lists of tensors that can be passed as separate
inputs/outputs to ONNX models.
"""

import torch
from typing import Any


class StateFlattener:
    """Handles flattening and unflattening of state dictionaries.
    
    State dictionaries have the structure:
    {
        "module_name": {
            "state_key": tensor,
            ...
        },
        ...
    }
    
    This class converts them to/from flat lists of tensors with metadata.
    """
    
    def __init__(self, state_schema: dict[str, dict[str, tuple[int, ...]]]):
        """Initialize with a schema describing the state structure.
        
        Args:
            state_schema: Dictionary mapping module names to their state schemas.
                Each module's schema maps state keys to tensor shapes (excluding batch dim).
                Example:
                {
                    "layers.0.self_attn": {
                        "current_end": (0,),  # 0-d tensor (scalar)
                        "cache": (2, 1, H, T, D),  # [2, batch, num_heads, seq_len, head_dim]
                    }
                }
        """
        self.state_schema = state_schema
        self._build_flat_schema()
    
    def _build_flat_schema(self):
        """Build a flat schema that maps indices to (module_name, state_key, shape)."""
        self.flat_schema = []
        for module_name, module_schema in sorted(self.state_schema.items()):
            for state_key, shape in sorted(module_schema.items()):
                self.flat_schema.append((module_name, state_key, shape))
    
    def flatten(self, state_dict: dict[str, dict[str, torch.Tensor]]) -> list[torch.Tensor]:
        """Flatten a state dictionary into a list of tensors.
        
        Args:
            state_dict: Nested state dictionary
            
        Returns:
            List of tensors in a deterministic order
        """
        result = []
        for module_name, state_key, _ in self.flat_schema:
            if module_name in state_dict and state_key in state_dict[module_name]:
                result.append(state_dict[module_name][state_key])
            else:
                # Create a zero tensor with the expected shape
                # This shouldn't happen in practice, but handle it gracefully
                raise ValueError(f"Missing state: {module_name}.{state_key}")
        return result
    
    def unflatten(self, flat_tensors: list[torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        """Unflatten a list of tensors back into a state dictionary.
        
        Args:
            flat_tensors: List of tensors in the same order as flatten()
            
        Returns:
            Nested state dictionary
        """
        if len(flat_tensors) != len(self.flat_schema):
            raise ValueError(
                f"Expected {len(self.flat_schema)} tensors, got {len(flat_tensors)}"
            )
        
        result = {}
        for (module_name, state_key, _), tensor in zip(self.flat_schema, flat_tensors):
            if module_name not in result:
                result[module_name] = {}
            result[module_name][state_key] = tensor
        
        return result


def build_flow_lm_state_schema(transformer: torch.nn.Module, batch_size: int, sequence_length: int) -> dict[str, dict[str, tuple[int, ...]]]:
    """Build state schema for FlowLM transformer.
    
    Args:
        transformer: The StreamingTransformer instance
        batch_size: Batch size for state initialization
        sequence_length: Maximum sequence length for KV cache
        
    Returns:
        State schema dictionary
    """
    from pocket_tts.modules.stateful_module import init_states
    
    # Initialize state to get actual shapes
    state = init_states(transformer, batch_size, sequence_length)
    
    # Build schema from actual state
    schema = {}
    for module_name, module_state in state.items():
        schema[module_name] = {}
        for state_key, tensor in module_state.items():
            # Store shape without batch dimension (batch is always first)
            shape = tuple(tensor.shape)
            schema[module_name][state_key] = shape
    
    return schema


def build_mimi_decoder_state_schema(decoder: torch.nn.Module, batch_size: int, sequence_length: int) -> dict[str, dict[str, tuple[int, ...]]]:
    """Build state schema for Mimi decoder.
    
    Args:
        decoder: The Mimi decoder module
        batch_size: Batch size for state initialization
        sequence_length: Maximum sequence length for KV cache
        
    Returns:
        State schema dictionary
    """
    from pocket_tts.modules.stateful_module import init_states
    
    # Initialize state to get actual shapes
    state = init_states(decoder, batch_size, sequence_length)
    
    # Build schema from actual state
    schema = {}
    for module_name, module_state in state.items():
        schema[module_name] = {}
        for state_key, tensor in module_state.items():
            # Store shape without batch dimension (batch is always first)
            shape = tuple(tensor.shape)
            schema[module_name][state_key] = shape
    
    return schema
