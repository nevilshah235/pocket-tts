# ONNX Conversion for Browser Deployment

This directory contains scripts and utilities for converting pocket-tts PyTorch models to ONNX format for browser deployment.

## Overview

The conversion process splits the TTS model into logical components:

1. **FlowLM Backbone** - Transformer that processes text and audio conditioning
2. **Flow Network** - MLP for flow matching (LSD decode)
3. **Mimi Encoder** - Encodes audio to latent space
4. **Mimi Decoder** - Decodes latents to audio
5. **Text Conditioner** - Tokenizes and embeds text

## Challenges

### Stateful Modules

The original models use stateful modules (KV caches, conv buffers) that maintain internal state. ONNX doesn't support mutable state well, so we need to:

1. **Flatten state to tensors** - Convert state dictionaries to flat tensor arrays
2. **Handle state in JavaScript** - Manage state updates in the browser runtime
3. **Export stateless forward passes** - Only export the computation, not state management

### Current Status

- ✅ **Model wrappers created** - Stateless wrappers for all components
- ✅ **Flow Network** - Ready for export (fully stateless)
- ✅ **Text Conditioner** - Ready for export (fully stateless)
- ✅ **Mimi Encoder** - Ready for export (fully stateless)
- ⚠️ **FlowLM Backbone** - Needs state flattening
- ⚠️ **Mimi Decoder** - Needs state flattening

## Usage

### Basic Export

```bash
# Export all components
python browser/onnx/convert_to_onnx.py --variant b6369a24 --output-dir browser/models/

# Export specific components
python browser/onnx/convert_to_onnx.py --components flow_net text_conditioner mimi_encoder
```

### Options

- `--variant`: Model variant (default: `b6369a24`)
- `--output-dir`: Output directory for ONNX models
- `--opset-version`: ONNX opset version (default: 17)
- `--components`: Which components to export (default: all)

## State Management Strategy

For stateful components, we use a two-phase approach:

1. **Export Phase**: Export stateless forward passes with state as inputs/outputs
2. **Runtime Phase**: JavaScript manages state updates between forward passes

### State Flattening

State dictionaries are flattened into tensors:
- KV caches → concatenated tensors
- Conv buffers → padding state tensors
- Positional offsets → integer tensors

The JavaScript runtime reconstructs state dictionaries from flattened tensors.

## Testing

Run the equivalence tests to verify refactoring:

```bash
# Test einops equivalence
uv run pytest browser/onnx/tests/test_equivalence.py -v

# Test module outputs
uv run pytest browser/onnx/tests/test_module_outputs.py -v

# Test refactored vs original
uv run pytest browser/onnx/tests/test_refactored_module_values.py -v
```

## Next Steps

1. Implement state flattening for FlowLM and Mimi decoder
2. Test ONNX export with actual model weights
3. Verify exported models load correctly in ONNX Runtime
4. Create JavaScript state management layer
