"""Stateless wrappers for stateful modules to enable ONNX export.

This module provides wrappers that convert stateful PyTorch modules into
stateless versions suitable for ONNX export. State is passed as inputs/outputs
rather than being stored internally.
"""

import torch
import torch.nn as nn
from typing import Any

from pocket_tts.modules.mimi_transformer import StreamingTransformer, ProjectedTransformer
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from in_the_browser.convert.state_utils import (
    StateFlattener,
    build_flow_lm_state_schema,
    build_mimi_decoder_state_schema,
)


class StatelessTransformerWrapper(nn.Module):
    """Wrapper for StreamingTransformer that makes it stateless for ONNX export.
    
    Instead of maintaining state internally, state is passed as input and returned as output.
    """

    def __init__(self, transformer: StreamingTransformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        x: torch.Tensor,
        model_state: dict[str, dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Forward pass with state as input/output.
        
        Args:
            x: Input tensor [B, T, D]
            model_state: Dictionary of module states
            
        Returns:
            output: Transformed tensor [B, T, D]
            model_state: Updated state dictionary
        """
        # The transformer modifies state in-place, so we need to work with a copy
        # For ONNX, we'll need to flatten the state into tensors
        output = self.transformer(x, model_state)
        return output, model_state


class StatelessFlowLMBackboneWrapper(nn.Module):
    """Wrapper for FlowLM backbone (transformer) that makes it stateless.
    
    This wrapper accepts flattened state tensors as separate inputs and returns
    flattened updated state tensors as separate outputs, making it suitable for ONNX export.
    """

    def __init__(self, flow_lm: FlowLMModel, batch_size: int = 1, sequence_length: int = 1000):
        super().__init__()
        self.flow_lm = flow_lm
        # Build state schema and flattener
        state_schema = build_flow_lm_state_schema(
            flow_lm.transformer, batch_size, sequence_length
        )
        self.state_flattener = StateFlattener(state_schema)
        self.num_state_tensors = len(self.state_flattener.flat_schema)

    def forward(
        self,
        input_: torch.Tensor,
        text_embeddings: torch.Tensor,
        sequence: torch.Tensor,
        *state_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass of FlowLM backbone with flattened state.
        
        Args:
            input_: Input latents [B, S, ldim]
            text_embeddings: Text conditioning [B, T_text, dim]
            sequence: Sequence tensor [B, S, ldim] (for shape info)
            *state_tensors: Flattened state tensors (variable number)
            
        Returns:
            output: Transformer output [B, S, dim]
            *updated_state_tensors: Flattened updated state tensors
        """
        # Unflatten state
        state_list = list(state_tensors)
        if len(state_list) != self.num_state_tensors:
            raise ValueError(
                f"Expected {self.num_state_tensors} state tensors, got {len(state_list)}"
            )
        model_state = self.state_flattener.unflatten(state_list)
        
        # Handle NaN values (BOS positions)
        sequence = torch.where(torch.isnan(sequence), self.flow_lm.bos_emb, sequence)
        input_linear_out = self.flow_lm.input_linear(sequence)
        
        # Concatenate text embeddings and input
        combined = torch.cat([text_embeddings, input_linear_out], dim=1)
        
        # Forward through transformer (modifies state in-place)
        transformer_out = self.flow_lm.transformer(combined, model_state)
        
        # Apply output norm
        if self.flow_lm.out_norm:
            transformer_out = self.flow_lm.out_norm(transformer_out)
        
        # Remove prefix (condition is prepended)
        transformer_out = transformer_out[:, -sequence.shape[1] :]
        
        # Flatten updated state
        updated_state_tensors = self.state_flattener.flatten(model_state)
        
        return (transformer_out, *updated_state_tensors)


class StatelessFlowNetworkWrapper(nn.Module):
    """Wrapper for flow network (MLP) - already stateless, but provides clean interface."""

    def __init__(self, flow_lm: FlowLMModel):
        super().__init__()
        self.flow_net = flow_lm.flow_net
        self.emb_std = flow_lm.emb_std
        self.emb_mean = flow_lm.emb_mean

    def forward(
        self,
        c: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of flow network.
        
        Args:
            c: Conditioning from transformer [B, dim]
            s: Start time tensor [B, 1] or scalar
            t: Target time tensor [B, 1] or scalar
            x: Input noise [B, ldim]
            
        Returns:
            output: Flow prediction [B, ldim]
        """
        return self.flow_net(c, s, t, x)


class StatelessMimiEncoderWrapper(nn.Module):
    """Wrapper for Mimi encoder - already stateless."""

    def __init__(self, mimi: MimiModel):
        super().__init__()
        self.mimi = mimi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent space.
        
        Args:
            x: Audio tensor [B, C, T]
            
        Returns:
            latents: Encoded latents [B, D, T']
        """
        return self.mimi.encode_to_latent(x)


class StatelessMimiDecoderWrapper(nn.Module):
    """Wrapper for Mimi decoder that makes it stateless.
    
    This wrapper accepts flattened state tensors as separate inputs and returns
    flattened updated state tensors as separate outputs, making it suitable for ONNX export.
    """

    def __init__(self, mimi: MimiModel, batch_size: int = 1, sequence_length: int = 1000):
        super().__init__()
        self.mimi = mimi
        # Build state schema from the full mimi model
        # This includes decoder_transformer, decoder (SEANet), and upsample if it exists
        from pocket_tts.modules.stateful_module import init_states
        full_state = init_states(mimi, batch_size, sequence_length)
        
        # Build schema from actual state
        state_schema = {}
        for module_name, module_state in full_state.items():
            state_schema[module_name] = {}
            for state_key, tensor in module_state.items():
                state_schema[module_name][state_key] = tuple(tensor.shape)
        
        self.state_flattener = StateFlattener(state_schema)
        self.num_state_tensors = len(self.state_flattener.flat_schema)

    def forward(
        self,
        latent: torch.Tensor,
        *state_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Decode latent to audio with flattened state.
        
        Args:
            latent: Latent tensor [B, D, T]
            *state_tensors: Flattened state tensors (variable number)
            
        Returns:
            audio: Decoded audio [B, C, T']
            *updated_state_tensors: Flattened updated state tensors
        """
        # Unflatten state
        state_list = list(state_tensors)
        if len(state_list) != self.num_state_tensors:
            raise ValueError(
                f"Expected {self.num_state_tensors} state tensors, got {len(state_list)}"
            )
        mimi_state = self.state_flattener.unflatten(state_list)
        
        # Quantize - quantizer expects [B, D, T] format (Conv1d operates on channel dim D)
        # The quantizer is a Conv1d, so it expects [B, C, T] where C is the channel dimension
        quantized = self.mimi.quantizer(latent)
        
        # Decode through transformer and SEANet (modifies state in-place)
        audio = self.mimi.decode_from_latent(quantized, mimi_state)
        
        # Flatten updated state
        updated_state_tensors = self.state_flattener.flatten(mimi_state)
        
        return (audio, *updated_state_tensors)


class StatelessTextConditionerWrapper(nn.Module):
    """Wrapper for text conditioner - already stateless."""

    def __init__(self, flow_lm: FlowLMModel):
        super().__init__()
        self.conditioner = flow_lm.conditioner

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Tokenize and embed text.
        
        Args:
            text_tokens: Token IDs [B, T]
            
        Returns:
            embeddings: Text embeddings [B, T, dim]
        """
        from pocket_tts.conditioners.base import TokenizedText
        
        return self.conditioner._get_condition(TokenizedText(text_tokens))
