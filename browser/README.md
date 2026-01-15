# Browser TTS Implementation

This directory contains the browser-based implementation of pocket-tts, using ONNX Runtime Web for model inference.

## Structure

```
browser/
├── index.html          # Web interface (placeholder for Phase 3)
├── onnx/               # ONNX conversion scripts and utilities
│   ├── convert_to_onnx.py    # Main conversion script
│   ├── model_wrappers.py     # Stateless model wrappers
│   ├── state_utils.py        # State flattening utilities
│   ├── validate_onnx.py      # Validation scripts
│   ├── test_all_models.py    # Test all exported models
│   └── tests/                # Conversion tests
├── models/             # Exported ONNX models
│   ├── flow_lm.onnx
│   ├── flow_net.onnx
│   ├── mimi_encoder.onnx
│   ├── mimi_decoder.onnx
│   └── text_conditioner.onnx
└── js/                 # JavaScript runtime (Phase 2)
```

## Phase 1: Model Conversion ✅

Phase 1 is complete. All 5 model components have been successfully converted to ONNX format:

- ✅ FlowLM Backbone (transformer with state flattening)
- ✅ Flow Network (MLP for flow matching)
- ✅ Mimi Encoder (audio → latent)
- ✅ Mimi Decoder (latent → audio with state flattening)
- ✅ Text Conditioner (tokenization and embedding)

See `onnx/README.md` for detailed information about the conversion process.

## Phase 2: JavaScript Runtime (Next)

The JavaScript runtime will handle:
- Model loading and inference
- State management (KV caches, conv buffers)
- Generation loop control flow
- Tokenizer implementation
- Audio processing utilities

## Phase 3: Web Interface (Future)

The web interface will provide:
- Text input and voice selection
- Real-time audio playback
- Voice cloning support
- Loading states and progress indicators

## Usage

### Converting Models

```bash
# Export all components
python browser/onnx/convert_to_onnx.py --variant b6369a24 --output-dir browser/models/

# Validate exported models
python browser/onnx/test_all_models.py --model-dir browser/models/
```

### Testing

```bash
# Run conversion tests
uv run pytest browser/onnx/tests/ -v
```

## Model Files

The ONNX model files are large (hundreds of MB) and should be:
- Stored using Git LFS, or
- Downloaded on-demand from a CDN, or
- Excluded from git (add to `.gitignore`)

See `onnx/README.md` for more details about the conversion process and state management strategy.
