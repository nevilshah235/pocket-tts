"""End-to-end integration test for ONNX models.

This test validates that all ONNX models work together correctly by running
a simplified version of the full TTS pipeline and comparing outputs with PyTorch.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F

from pocket_tts import TTSModel
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.modules.stateful_module import init_states, increment_steps

# Import state utilities
import sys
from pathlib import Path as PathLib

# Add convert directory to path for imports
convert_dir = PathLib(__file__).parent.parent
sys.path.insert(0, str(convert_dir))

from state_utils import StateFlattener, build_flow_lm_state_schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXPipeline:
    """Wrapper for running the full TTS pipeline using ONNX models."""

    def __init__(self, model_dir: Path, model: TTSModel):
        """Initialize ONNX pipeline with all models.
        
        Args:
            model_dir: Directory containing ONNX model files
            model: PyTorch TTSModel for reference (weights, config, etc.)
        """
        self.model = model
        self.model_dir = Path(model_dir)
        
        # Load all ONNX models
        logger.info("Loading ONNX models...")
        self.text_conditioner_session = ort.InferenceSession(
            str(self.model_dir / "text_conditioner.onnx")
        )
        self.mimi_encoder_session = ort.InferenceSession(
            str(self.model_dir / "mimi_encoder.onnx")
        )
        self.flow_lm_backbone_session = ort.InferenceSession(
            str(self.model_dir / "flow_lm.onnx")
        )
        self.flow_net_session = ort.InferenceSession(
            str(self.model_dir / "flow_net.onnx")
        )
        self.mimi_decoder_session = ort.InferenceSession(
            str(self.model_dir / "mimi_decoder.onnx")
        )
        
        # Build state flatteners
        batch_size = 1
        sequence_length = 1000
        
        flow_lm_schema = build_flow_lm_state_schema(
            model.flow_lm.transformer, batch_size, sequence_length
        )
        self.flow_lm_flattener = StateFlattener(flow_lm_schema)
        
        mimi_state = init_states(model.mimi, batch_size, sequence_length)
        mimi_schema = {}
        for module_name, module_state in mimi_state.items():
            mimi_schema[module_name] = {}
            for state_key, tensor in module_state.items():
                mimi_schema[module_name][state_key] = tuple(tensor.shape)
        self.mimi_flattener = StateFlattener(mimi_schema)
        
        # Store model parameters
        self.ldim = model.flow_lm.ldim
        self.dim = model.flow_lm.dim
        self.temp = model.temp
        self.lsd_decode_steps = model.lsd_decode_steps
        self.noise_clamp = model.noise_clamp
        self.eos_threshold = model.eos_threshold
        self.emb_std = model.flow_lm.emb_std
        self.emb_mean = model.flow_lm.emb_mean
        self.bos_emb = model.flow_lm.bos_emb
        self.speaker_proj_weight = model.flow_lm.speaker_proj_weight
        
        # Get input/output names for stateful models
        self.flow_lm_input_names = [inp.name for inp in self.flow_lm_backbone_session.get_inputs()]
        self.flow_lm_output_names = [out.name for out in self.flow_lm_backbone_session.get_outputs()]
        self.mimi_decoder_input_names = [inp.name for inp in self.mimi_decoder_session.get_inputs()]
        self.mimi_decoder_output_names = [out.name for out in self.mimi_decoder_session.get_outputs()]

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to conditioning embeddings using ONNX mimi encoder.
        
        Args:
            audio: Audio tensor [B, C, T]
            
        Returns:
            Conditioning embeddings [B, T_cond, dim]
        """
        # Run ONNX encoder
        inputs = {"audio": audio.numpy()}
        outputs = self.mimi_encoder_session.run(None, inputs)
        latents = torch.from_numpy(outputs[0])  # [B, D, T']
        
        # Transpose and project (same as PyTorch)
        latents = latents.transpose(-1, -2).to(torch.float32)  # [B, T', D]
        conditioning = F.linear(latents, self.speaker_proj_weight)  # [B, T', dim]
        return conditioning

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using the PyTorch tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            Token IDs [B, T]
        """
        tokens = self.model.flow_lm.conditioner.tokenizer(text)
        return tokens.tokens  # [B, T]

    def get_text_embeddings(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Get text embeddings using ONNX text conditioner.
        
        Args:
            text_tokens: Token IDs [B, T]
            
        Returns:
            Text embeddings [B, T, dim]
        """
        inputs = {"text_tokens": text_tokens.numpy().astype(np.int64)}
        outputs = self.text_conditioner_session.run(None, inputs)
        return torch.from_numpy(outputs[0])

    def run_flow_lm_backbone(
        self,
        input_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        audio_conditioning: torch.Tensor,
        flow_lm_state_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run FlowLM backbone using ONNX.
        
        Args:
            input_latents: Input latents [B, S, ldim]
            text_embeddings: Text embeddings [B, T_text, dim]
            audio_conditioning: Audio conditioning [B, T_audio, dim]
            flow_lm_state_tensors: Flattened state tensors
            
        Returns:
            transformer_out: Transformer output [B, S, dim]
            updated_state_tensors: Updated flattened state tensors
        """
        # Combine text and audio conditioning
        combined_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
        
        # Prepare sequence (handle NaN for BOS)
        sequence = torch.where(torch.isnan(input_latents), self.bos_emb, input_latents)
        
        # Build input dict
        inputs = {
            "input": input_latents.numpy(),
            "text_embeddings": combined_embeddings.numpy(),
            "sequence": sequence.numpy(),
        }
        
        # Add state tensors
        for i, state_tensor in enumerate(flow_lm_state_tensors):
            state_name = self.flow_lm_input_names[3 + i]  # Skip first 3 regular inputs
            inputs[state_name] = state_tensor.numpy()
        
        # Run ONNX model
        outputs = self.flow_lm_backbone_session.run(None, inputs)
        
        # Extract transformer output and updated states
        transformer_out = torch.from_numpy(outputs[0])  # [B, S, dim]
        updated_state_tensors = [torch.from_numpy(out) for out in outputs[1:]]
        
        return transformer_out, updated_state_tensors

    def run_flow_net(
        self, conditioning: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Run flow network using ONNX.
        
        Args:
            conditioning: Conditioning from transformer [B, dim]
            s: Start time [B, 1]
            t: Target time [B, 1]
            x: Input noise [B, ldim]
            
        Returns:
            Flow output [B, ldim]
        """
        inputs = {
            "conditioning": conditioning.numpy(),
            "start_time": s.numpy(),
            "target_time": t.numpy(),
            "input_noise": x.numpy(),
        }
        outputs = self.flow_net_session.run(None, inputs)
        return torch.from_numpy(outputs[0])

    def lsd_decode_onnx(
        self, conditioning: torch.Tensor, x_0: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Run LSD decode using ONNX flow network.
        
        Args:
            conditioning: Conditioning from transformer [B, dim]
            x_0: Starting noise [B, ldim]
            num_steps: Number of decode steps
            
        Returns:
            Decoded latent [B, ldim]
        """
        current = x_0
        for i in range(num_steps):
            s = i / num_steps
            t = (i + 1) / num_steps
            s_tensor = s * torch.ones_like(x_0[..., :1])
            t_tensor = t * torch.ones_like(x_0[..., :1])
            flow_dir = self.run_flow_net(conditioning, s_tensor, t_tensor, current)
            current = current + flow_dir / num_steps
        return current

    def compute_eos(self, transformer_out: torch.Tensor) -> torch.Tensor:
        """Compute EOS prediction using PyTorch (not exported to ONNX yet).
        
        Args:
            transformer_out: Transformer output [B, dim]
            
        Returns:
            EOS prediction [B, 1] (boolean)
        """
        # Use PyTorch model for EOS (not exported to ONNX)
        out_eos = self.model.flow_lm.out_eos(transformer_out) > self.eos_threshold
        return out_eos

    def decode_audio(
        self, latent: torch.Tensor, mimi_state_tensors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode latent to audio using ONNX mimi decoder.
        
        Args:
            latent: Latent tensor [B, D, T] (after quantizer dimension)
            mimi_state_tensors: Flattened state tensors
            
        Returns:
            audio: Decoded audio [B, C, T']
            updated_state_tensors: Updated flattened state tensors
        """
        # Prepare inputs
        inputs = {"latent": latent.numpy()}
        
        # Add state tensors
        for i, state_tensor in enumerate(mimi_state_tensors):
            state_name = self.mimi_decoder_input_names[1 + i]  # Skip first input (latent)
            inputs[state_name] = state_tensor.numpy()
        
        # Run ONNX model
        outputs = self.mimi_decoder_session.run(None, inputs)
        
        # Extract audio and updated states
        audio = torch.from_numpy(outputs[0])  # [B, C, T']
        updated_state_tensors = [torch.from_numpy(out) for out in outputs[1:]]
        
        return audio, updated_state_tensors

    def generate_step(
        self,
        input_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        audio_conditioning: torch.Tensor,
        flow_lm_state_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Generate one step of latents.
        
        Args:
            input_latents: Input latents [B, S, ldim] (S=1 for streaming)
            text_embeddings: Text embeddings [B, T_text, dim]
            audio_conditioning: Audio conditioning [B, T_audio, dim]
            flow_lm_state_tensors: Flattened FlowLM state tensors
            
        Returns:
            next_latent: Next latent [B, ldim]
            is_eos: EOS prediction [B, 1]
            updated_flow_lm_state_tensors: Updated state tensors
        """
        # Run backbone
        transformer_out, updated_state_tensors = self.run_flow_lm_backbone(
            input_latents, text_embeddings, audio_conditioning, flow_lm_state_tensors
        )
        
        # Get last timestep output
        transformer_out_last = transformer_out[:, -1]  # [B, dim]
        
        # Compute EOS
        is_eos = self.compute_eos(transformer_out_last)
        
        # Sample noise
        noise_shape = transformer_out_last.shape[:-1] + (self.ldim,)
        std = self.temp**0.5
        noise = torch.empty(noise_shape, dtype=transformer_out_last.dtype)
        if self.noise_clamp is None:
            torch.nn.init.normal_(noise, mean=0.0, std=std)
        else:
            torch.nn.init.trunc_normal_(
                noise, mean=0.0, std=std, a=-self.noise_clamp, b=self.noise_clamp
            )
        
        # Run LSD decode
        next_latent = self.lsd_decode_onnx(transformer_out_last, noise, self.lsd_decode_steps)
        
        return next_latent, is_eos, updated_state_tensors


def test_onnx_integration():
    """Test full ONNX pipeline against PyTorch version."""
    # Skip if models aren't available
    try:
        from pocket_tts import TTSModel
    except ImportError:
        pytest.skip("pocket_tts not available")
    
    # Check if ONNX models exist
    # Default output directory is in_the_browser/models (relative to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    model_dir = project_root / "in_the_browser" / "models"
    if not model_dir.exists():
        pytest.skip(f"ONNX models not found at {model_dir}. Run convert_to_onnx.py first.")
    
    required_models = [
        "text_conditioner.onnx",
        "mimi_encoder.onnx",
        "flow_lm.onnx",
        "flow_net.onnx",
        "mimi_decoder.onnx",
    ]
    
    for model_file in required_models:
        if not (model_dir / model_file).exists():
            pytest.skip(f"Required ONNX model not found: {model_file}")
    
    logger.info("Loading PyTorch model...")
    pytorch_model = TTSModel.load_model()
    pytorch_model.eval()
    
    logger.info("Initializing ONNX pipeline...")
    onnx_pipeline = ONNXPipeline(model_dir, pytorch_model)
    
    # Test parameters
    test_text = "Hello world"
    test_audio_duration = 0.5  # seconds
    
    # Prepare text
    logger.info("Tokenizing text...")
    text_tokens = onnx_pipeline.tokenize_text(test_text)
    
    # Get text embeddings (ONNX)
    logger.info("Getting text embeddings (ONNX)...")
    text_embeddings_onnx = onnx_pipeline.get_text_embeddings(text_tokens)
    
    # Get text embeddings (PyTorch) for comparison
    logger.info("Getting text embeddings (PyTorch)...")
    with torch.no_grad():
        text_embeddings_pytorch = pytorch_model.flow_lm.conditioner(
            TokenizedText(text_tokens)
        )
    
    # Compare text embeddings
    max_diff = torch.abs(text_embeddings_onnx - text_embeddings_pytorch).max().item()
    mean_diff = torch.abs(text_embeddings_onnx - text_embeddings_pytorch).mean().item()
    logger.info(f"Text embeddings comparison:")
    logger.info(f"  Max difference: {max_diff:.2e}")
    logger.info(f"  Mean difference: {mean_diff:.2e}")
    assert torch.allclose(
        text_embeddings_onnx, text_embeddings_pytorch, atol=1e-5
    ), f"Text embeddings differ: max_diff={max_diff:.2e}"
    
    # Prepare audio prompt
    logger.info("Preparing audio prompt...")
    sample_rate = pytorch_model.config.mimi.sample_rate
    channels = pytorch_model.config.mimi.channels
    num_samples = int(sample_rate * test_audio_duration)
    test_audio = torch.randn(1, channels, num_samples)
    
    # Encode audio (ONNX)
    logger.info("Encoding audio (ONNX)...")
    audio_conditioning_onnx = onnx_pipeline.encode_audio(test_audio)
    
    # Encode audio (PyTorch) for comparison
    logger.info("Encoding audio (PyTorch)...")
    with torch.no_grad():
        encoded_pytorch = pytorch_model.mimi.encode_to_latent(test_audio)
        latents_pytorch = encoded_pytorch.transpose(-1, -2).to(torch.float32)
        audio_conditioning_pytorch = F.linear(
            latents_pytorch, pytorch_model.flow_lm.speaker_proj_weight
        )
    
    # Compare audio conditioning
    max_diff = torch.abs(audio_conditioning_onnx - audio_conditioning_pytorch).max().item()
    mean_diff = torch.abs(audio_conditioning_onnx - audio_conditioning_pytorch).mean().item()
    logger.info(f"Audio conditioning comparison:")
    logger.info(f"  Max difference: {max_diff:.2e}")
    logger.info(f"  Mean difference: {mean_diff:.2e}")
    assert torch.allclose(
        audio_conditioning_onnx, audio_conditioning_pytorch, atol=1e-4
    ), f"Audio conditioning differs: max_diff={max_diff:.2e}"
    
    # Initialize states
    logger.info("Initializing states...")
    batch_size = 1
    sequence_length = 1000
    
    # FlowLM state
    flow_lm_state_pytorch = init_states(pytorch_model.flow_lm.transformer, batch_size, sequence_length)
    flow_lm_state_tensors = onnx_pipeline.flow_lm_flattener.flatten(flow_lm_state_pytorch)
    
    # Mimi decoder state
    mimi_state_pytorch = init_states(pytorch_model.mimi, batch_size, sequence_length)
    mimi_state_tensors = onnx_pipeline.mimi_flattener.flatten(mimi_state_pytorch)
    
    # Test a few generation steps
    logger.info("Running generation steps...")
    num_steps = 3
    
    # PyTorch version
    flow_lm_state_pytorch_copy = copy.deepcopy(flow_lm_state_pytorch)
    mimi_state_pytorch_copy = copy.deepcopy(mimi_state_pytorch)
    
    pytorch_latents = []
    onnx_latents = []
    
    # Initial input (BOS)
    input_latents = torch.full(
        (1, 1, pytorch_model.flow_lm.ldim),
        fill_value=float("NaN"),
        dtype=pytorch_model.flow_lm.dtype,
    )
    
    # First step: include text tokens
    # Subsequent steps: empty text tokens
    empty_text_tokens = torch.zeros((1, 0), dtype=torch.int64)
    
    for step in range(num_steps):
        logger.info(f"Generation step {step + 1}/{num_steps}...")
        
        # Use text tokens only on first step
        current_text_tokens = text_tokens if step == 0 else empty_text_tokens
        current_text_embeddings = text_embeddings_onnx if step == 0 else torch.zeros(
            (1, 0, pytorch_model.flow_lm.dim), dtype=text_embeddings_onnx.dtype
        )
        
        # ONNX version
        next_latent_onnx, is_eos_onnx, flow_lm_state_tensors = onnx_pipeline.generate_step(
            input_latents,
            current_text_embeddings,
            audio_conditioning_onnx,
            flow_lm_state_tensors,
        )
        onnx_latents.append(next_latent_onnx.clone())
        
        # PyTorch version
        with torch.no_grad():
            output_embeddings_pytorch, is_eos_pytorch = pytorch_model._run_flow_lm(
                model_state=flow_lm_state_pytorch_copy,
                text_tokens=current_text_tokens,
                backbone_input_latents=input_latents,
                audio_conditioning=audio_conditioning_pytorch,
            )
            next_latent_pytorch = output_embeddings_pytorch[0, 0]  # [ldim]
            pytorch_latents.append(next_latent_pytorch.clone())
            
            # Increment step (PyTorch)
            increment_by = current_text_tokens.shape[1] + input_latents.shape[1] + audio_conditioning_pytorch.shape[1]
            increment_steps(
                pytorch_model.flow_lm, flow_lm_state_pytorch_copy, increment=increment_by
            )
        
        # Compare latents
        max_diff = torch.abs(next_latent_onnx - next_latent_pytorch).max().item()
        mean_diff = torch.abs(next_latent_onnx - next_latent_pytorch).mean().item()
        logger.info(f"  Latent comparison:")
        logger.info(f"    Max difference: {max_diff:.2e}")
        logger.info(f"    Mean difference: {mean_diff:.2e}")
        
        # Note: We expect some differences due to numerical precision and state handling
        # Using a reasonable tolerance
        assert torch.allclose(
            next_latent_onnx, next_latent_pytorch, atol=1e-3, rtol=1e-3
        ), f"Latent differs at step {step}: max_diff={max_diff:.2e}"
        
        # Update input for next step
        input_latents = next_latent_onnx[:, None, :]  # [B, 1, ldim]
    
    # Test audio decoding for one latent
    logger.info("Testing audio decoding...")
    test_latent = onnx_latents[0]  # [B, ldim]
    
    # Prepare latent for decoder (same as in _decode_audio_worker)
    # The latent from flow_lm is [B, ldim], we need [B, ldim, 1] for quantizer
    mimi_decoding_input = test_latent * onnx_pipeline.emb_std + onnx_pipeline.emb_mean
    # Expand to [B, 1, ldim] then transpose to [B, ldim, 1]
    test_latent_expanded = test_latent[:, None, :]  # [B, 1, ldim]
    mimi_decoding_input_expanded = mimi_decoding_input[:, None, :]  # [B, 1, ldim]
    transposed = mimi_decoding_input_expanded.transpose(-1, -2)  # [B, ldim, 1]
    
    # ONNX decoder (expects [B, quantizer_dim, T] where quantizer_dim = ldim)
    audio_onnx, mimi_state_tensors = onnx_pipeline.decode_audio(transposed, mimi_state_tensors)
    
    # PyTorch decoder
    with torch.no_grad():
        quantized_pytorch = pytorch_model.mimi.quantizer(transposed)
        audio_pytorch = pytorch_model.mimi.decode_from_latent(
            quantized_pytorch, mimi_state_pytorch_copy
        )
        increment_steps(pytorch_model.mimi, mimi_state_pytorch_copy, increment=16)
    
    # Compare audio
    max_diff = torch.abs(audio_onnx - audio_pytorch).max().item()
    mean_diff = torch.abs(audio_onnx - audio_pytorch).mean().item()
    logger.info(f"Audio decoding comparison:")
    logger.info(f"  Max difference: {max_diff:.2e}")
    logger.info(f"  Mean difference: {mean_diff:.2e}")
    
    # Audio may have larger differences due to cumulative state effects
    assert torch.allclose(
        audio_onnx, audio_pytorch, atol=1e-2, rtol=1e-2
    ), f"Audio differs: max_diff={max_diff:.2e}"
    
    logger.info("✓ ONNX integration test passed!")


if __name__ == "__main__":
    test_onnx_integration()
