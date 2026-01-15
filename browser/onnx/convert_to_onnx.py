"""Convert PyTorch models to ONNX format for browser deployment.

This script exports the pocket-tts models to ONNX format, splitting them into
logical components that can be loaded and run in the browser using ONNX Runtime Web.

Usage:
    python convert_to_onnx.py --variant b6369a24 --output-dir browser/models/
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.onnx

import sys
from pathlib import Path

# Add convert directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pocket_tts import TTSModel
from pocket_tts.utils.config import load_config

from model_wrappers import (
    StatelessFlowLMBackboneWrapper,
    StatelessFlowNetworkWrapper,
    StatelessMimiEncoderWrapper,
    StatelessMimiDecoderWrapper,
    StatelessTextConditionerWrapper,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_flow_lm_backbone(
    model: TTSModel, output_path: Path, opset_version: int = 17
) -> None:
    """Export FlowLM transformer backbone to ONNX.
    
    Args:
        model: Loaded TTSModel instance
        output_path: Path to save ONNX model
        opset_version: ONNX opset version to use
    """
    logger.info("Exporting FlowLM backbone...")
    
    batch_size = 1
    seq_len = 10
    text_len = 5
    sequence_length = 1000  # Max sequence length for KV cache
    
    # Initialize wrapper with state schema
    wrapper = StatelessFlowLMBackboneWrapper(
        model.flow_lm, batch_size=batch_size, sequence_length=sequence_length
    )
    wrapper.eval()
    
    # Create dummy inputs
    dim = model.flow_lm.dim
    ldim = model.flow_lm.ldim
    
    input_ = torch.randn(batch_size, seq_len, ldim)
    text_embeddings = torch.randn(batch_size, text_len, dim)
    sequence = torch.randn(batch_size, seq_len, ldim)
    
    # Initialize state and flatten it
    from pocket_tts.modules.stateful_module import init_states
    
    model_state = init_states(model.flow_lm.transformer, batch_size, sequence_length)
    state_tensors = wrapper.state_flattener.flatten(model_state)
    
    # Build input/output names
    input_names = ["input", "text_embeddings", "sequence"]
    output_names = ["output"]
    
    # Add state tensor names
    for i, (module_name, state_key, _) in enumerate(wrapper.state_flattener.flat_schema):
        state_name = f"state_{module_name.replace('.', '_')}_{state_key}"
        input_names.append(state_name)
        output_names.append(f"updated_{state_name}")
    
    try:
        # Prepare inputs: regular inputs + state tensors
        export_inputs = (input_, text_embeddings, sequence, *state_tensors)
        
        # For now, don't use dynamic_axes to avoid compatibility issues
        # The model will work with fixed sequence lengths at export time
        # Dynamic shapes can be added later if needed
        torch.onnx.export(
            wrapper,
            export_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
            # dynamic_axes=dynamic_axes,  # Disabled for now
        )
        logger.info(f"✓ Exported FlowLM backbone to {output_path}")
        logger.info(f"  State tensors: {len(state_tensors)} inputs, {len(state_tensors)} outputs")
    except Exception as e:
        logger.error(f"Could not export FlowLM backbone: {e}")
        import traceback
        traceback.print_exc()
        raise


def export_flow_network(
    model: TTSModel, output_path: Path, opset_version: int = 17
) -> None:
    """Export flow network (MLP) to ONNX.
    
    Args:
        model: Loaded TTSModel instance
        output_path: Path to save ONNX model
        opset_version: ONNX opset version to use
    """
    logger.info("Exporting flow network...")
    
    wrapper = StatelessFlowNetworkWrapper(model.flow_lm)
    wrapper.eval()
    
    # Create dummy inputs
    batch_size = 1
    dim = model.flow_lm.dim
    ldim = model.flow_lm.ldim
    
    c = torch.randn(batch_size, dim)  # Conditioning
    s = torch.randn(batch_size, 1)  # Start time
    t = torch.randn(batch_size, 1)  # Target time
    x = torch.randn(batch_size, ldim)  # Input noise
    
    torch.onnx.export(
        wrapper,
        (c, s, t, x),
        str(output_path),
        input_names=["conditioning", "start_time", "target_time", "input_noise"],
        output_names=["flow_output"],
        opset_version=opset_version,
        do_constant_folding=False,  # Disable to avoid segfaults on some systems
        verbose=False,
        dynamic_axes={
            "conditioning": {0: "batch_size"},
            "start_time": {0: "batch_size"},
            "target_time": {0: "batch_size"},
            "input_noise": {0: "batch_size"},
            "flow_output": {0: "batch_size"},
        },
    )
    logger.info(f"✓ Exported flow network to {output_path}")


def export_mimi_encoder(
    model: TTSModel, output_path: Path, opset_version: int = 17
) -> None:
    """Export Mimi encoder to ONNX.
    
    Args:
        model: Loaded TTSModel instance
        output_path: Path to save ONNX model
        opset_version: ONNX opset version to use
    """
    logger.info("Exporting Mimi encoder...")
    
    wrapper = StatelessMimiEncoderWrapper(model.mimi)
    wrapper.eval()
    
    # Create dummy input
    batch_size = 1
    channels = model.mimi.channels
    sample_rate = model.mimi.sample_rate
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    
    x = torch.randn(batch_size, channels, num_samples)
    
    torch.onnx.export(
        wrapper,
        x,
        str(output_path),
        input_names=["audio"],
        output_names=["latents"],
        opset_version=opset_version,
        do_constant_folding=False,  # Disable to avoid segfaults on some systems
        verbose=False,
        dynamic_axes={
            "audio": {2: "num_samples"},
            "latents": {2: "latent_length"},
        },
    )
    logger.info(f"✓ Exported Mimi encoder to {output_path}")


def export_mimi_decoder(
    model: TTSModel, output_path: Path, opset_version: int = 17
) -> None:
    """Export Mimi decoder to ONNX.
    
    Args:
        model: Loaded TTSModel instance
        output_path: Path to save ONNX model
        opset_version: ONNX opset version to use
    """
    logger.info("Exporting Mimi decoder...")
    
    batch_size = 1
    sequence_length = 1000  # Max sequence length for KV cache
    
    # Initialize wrapper with state schema
    wrapper = StatelessMimiDecoderWrapper(
        model.mimi, batch_size=batch_size, sequence_length=sequence_length
    )
    wrapper.eval()
    
    # Create dummy inputs
    # The quantizer expects input with dimension channels (not output_dimension)
    quantizer_dimension = model.mimi.quantizer.dimension
    num_frames = 10
    
    latent = torch.randn(batch_size, quantizer_dimension, num_frames)
    
    # Initialize state and flatten it
    from pocket_tts.modules.stateful_module import init_states
    
    # Get state from the full mimi model (includes decoder, upsample, etc.)
    mimi_state = init_states(model.mimi, batch_size, sequence_length)
    state_tensors = wrapper.state_flattener.flatten(mimi_state)
    
    # Build input/output names
    input_names = ["latent"]
    output_names = ["audio"]
    
    # Add state tensor names
    for i, (module_name, state_key, _) in enumerate(wrapper.state_flattener.flat_schema):
        state_name = f"state_{module_name.replace('.', '_')}_{state_key}"
        input_names.append(state_name)
        output_names.append(f"updated_{state_name}")
    
    try:
        # Prepare inputs: latent + state tensors
        export_inputs = (latent, *state_tensors)
        
        torch.onnx.export(
            wrapper,
            export_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=True,
            # dynamic_axes disabled for now (same as flow_lm)
        )
        logger.info(f"✓ Exported Mimi decoder to {output_path}")
        logger.info(f"  State tensors: {len(state_tensors)} inputs, {len(state_tensors)} outputs")
    except Exception as e:
        logger.error(f"Could not export Mimi decoder: {e}")
        import traceback
        traceback.print_exc()
        raise


def export_text_conditioner(
    model: TTSModel, output_path: Path, opset_version: int = 17
) -> None:
    """Export text conditioner to ONNX.
    
    Args:
        model: Loaded TTSModel instance
        output_path: Path to save ONNX model
        opset_version: ONNX opset version to use
    """
    logger.info("Exporting text conditioner...")
    
    wrapper = StatelessTextConditionerWrapper(model.flow_lm)
    wrapper.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 10
    vocab_size = model.flow_lm.conditioner.embed.num_embeddings
    
    text_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    torch.onnx.export(
        wrapper,
        text_tokens,
        str(output_path),
        input_names=["text_tokens"],
        output_names=["text_embeddings"],
        opset_version=opset_version,
        do_constant_folding=False,  # Disable to avoid segfaults on some systems
        verbose=False,
        dynamic_axes={
            "text_tokens": {1: "seq_len"},
            "text_embeddings": {1: "seq_len"},
        },
    )
    logger.info(f"✓ Exported text conditioner to {output_path}")


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert pocket-tts models to ONNX")
    parser.add_argument(
        "--variant",
        type=str,
        default="b6369a24",
        help="Model variant to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="browser/models",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version to use (default: 18, recommended for browser)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["all", "flow_lm", "flow_net", "mimi_encoder", "mimi_decoder", "text_conditioner"],
        default=["all"],
        help="Which components to export",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model variant: {args.variant}")
    model = TTSModel.load_model(variant=args.variant)
    model.eval()
    
    components = args.components
    if "all" in components:
        components = ["flow_lm", "flow_net", "mimi_encoder", "mimi_decoder", "text_conditioner"]
    
    if "flow_lm" in components:
        export_flow_lm_backbone(
            model, output_dir / "flow_lm.onnx", args.opset_version
        )
    
    if "flow_net" in components:
        export_flow_network(
            model, output_dir / "flow_net.onnx", args.opset_version
        )
    
    if "mimi_encoder" in components:
        export_mimi_encoder(
            model, output_dir / "mimi_encoder.onnx", args.opset_version
        )
    
    if "mimi_decoder" in components:
        export_mimi_decoder(
            model, output_dir / "mimi_decoder.onnx", args.opset_version
        )
    
    if "text_conditioner" in components:
        export_text_conditioner(
            model, output_dir / "text_conditioner.onnx", args.opset_version
        )
    
    logger.info("✓ Conversion complete!")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
