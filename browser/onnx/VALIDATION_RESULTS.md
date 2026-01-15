# ONNX Conversion Validation Results

## Test Summary

Date: 2025-01-15

### Components Tested

#### 1. Flow Network ✅
- **Status**: Export and validation successful
- **File Size**: 349 KB
- **ONNX Validation**: ✓ Model is valid
- **Output Comparison**:
  - Shape match: ✓
  - Max difference: 1.19e-06
  - Mean difference: 3.54e-07
  - All close (atol=1e-5): ✓
- **Conclusion**: ONNX model produces equivalent outputs to PyTorch

#### 2. Text Conditioner ✅
- **Status**: Export and validation successful
- **File Size**: 2 KB
- **ONNX Validation**: ✓ Model is valid
- **Output Comparison**:
  - Shape match: ✓
  - Max difference: 0.00e+00 (exact match!)
  - Mean difference: 0.00e+00
  - All close (atol=1e-5): ✓
- **Conclusion**: ONNX model produces identical outputs to PyTorch

## Export Configuration

Working configuration:
- **ONNX Opset**: 18 (recommended, auto-upgraded from 17)
- **Constant Folding**: Disabled (prevents segfaults on some systems)
- **Dynamic Axes**: Enabled for batch and sequence dimensions

## Validation Process

1. **ONNX Model Validation**: Check model structure and validity using `onnx.checker.check_model()`
2. **Output Comparison**: Run same inputs through both PyTorch and ONNX models
3. **Numerical Verification**: Compare outputs using `torch.allclose()` with tolerance 1e-5

## Test Commands

```bash
# Export flow network
uv run python -m in_the_browser.convert.convert_to_onnx \
    --components flow_net \
    --output-dir /tmp/test_onnx

# Validate flow network
uv run python -m in_the_browser.convert.validate_onnx \
    --model-path /tmp/test_onnx/flow_net.onnx \
    --component flow_net

# Export text conditioner
uv run python -m in_the_browser.convert.convert_to_onnx \
    --components text_conditioner \
    --output-dir /tmp/test_onnx

# Validate text conditioner
uv run python -m in_the_browser.convert.validate_onnx \
    --model-path /tmp/test_onnx/text_conditioner.onnx \
    --component text_conditioner
```

## Next Steps

- [ ] Export and validate Mimi encoder
- [ ] Implement state flattening for FlowLM backbone
- [ ] Implement state flattening for Mimi decoder
- [ ] Test full pipeline with all components
