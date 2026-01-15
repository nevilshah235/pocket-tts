"""Test all Phase 1 exported ONNX models.

This script validates all 5 Phase 1 ONNX models:
1. flow_net
2. text_conditioner
3. mimi_encoder
4. flow_lm (FlowLM backbone)
5. mimi_decoder

Usage:
    python test_all_models.py [--model-dir browser/models] [--variant b6369a24]
"""

import argparse
import logging
import sys
from pathlib import Path

from browser.onnx.validate_onnx import (
    validate_flow_network,
    validate_text_conditioner,
    validate_mimi_encoder,
    validate_flow_lm_backbone,
    validate_mimi_decoder,
)
from pocket_tts import TTSModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Test all Phase 1 ONNX models."""
    parser = argparse.ArgumentParser(
        description="Test all Phase 1 exported ONNX models"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="browser/models",
        help="Directory containing ONNX model files",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="b6369a24",
        help="Model variant to use for comparison",
    )
    parser.add_argument(
        "--use-realistic-inputs",
        action="store_true",
        help="Use realistic inputs from actual pipeline (single frames, real text/audio) "
             "instead of random inputs. Produces smaller differences and tests real usage patterns. "
             "Only applies to flow_lm and mimi_decoder components.",
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.error("Please export models first using convert_to_onnx.py")
        return 1
    
    # Define all Phase 1 models
    models = [
        ("flow_net", "flow_net.onnx", validate_flow_network),
        ("text_conditioner", "text_conditioner.onnx", validate_text_conditioner),
        ("mimi_encoder", "mimi_encoder.onnx", validate_mimi_encoder),
        ("flow_lm", "flow_lm.onnx", validate_flow_lm_backbone),
        ("mimi_decoder", "mimi_decoder.onnx", validate_mimi_decoder),
    ]
    
    logger.info(f"Loading PyTorch model variant: {args.variant}")
    try:
        model = TTSModel.load_model(variant=args.variant)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        return 1
    
    logger.info("=" * 70)
    logger.info("Testing all Phase 1 ONNX models")
    logger.info("=" * 70)
    
    results = {}
    for model_name, model_file, validate_func in models:
        model_path = model_dir / model_file
        
        if not model_path.exists():
            logger.warning(f"⚠ Model not found: {model_path}")
            logger.warning(f"  Skipping {model_name} validation")
            results[model_name] = None
            continue
        
        logger.info("")
        logger.info("-" * 70)
        logger.info(f"Testing {model_name}")
        logger.info("-" * 70)
        
        try:
            # Pass use_realistic_inputs only to flow_lm and mimi_decoder
            if model_name in ("flow_lm", "mimi_decoder"):
                success = validate_func(model_path, model, use_realistic_inputs=args.use_realistic_inputs)
            else:
                success = validate_func(model_path, model)
            results[model_name] = success
        except Exception as e:
            logger.error(f"✗ Error validating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = False
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    all_passed = True
    for model_name, result in results.items():
        if result is None:
            status = "⚠ SKIPPED (file not found)"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        logger.info(f"  {model_name:20s}: {status}")
    
    logger.info("=" * 70)
    
    if all_passed and all(r is not None for r in results.values()):
        logger.info("✓ All Phase 1 models validated successfully!")
        return 0
    elif any(r is None for r in results.values()):
        logger.warning("⚠ Some models were skipped (files not found)")
        logger.warning("  Run convert_to_onnx.py to export missing models")
        return 1
    else:
        logger.error("✗ Some models failed validation")
        return 1


if __name__ == "__main__":
    exit(main())
