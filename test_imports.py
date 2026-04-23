#!/usr/bin/env python3
"""Quick smoke test for all rebuilt modules."""
import sys
sys.path.insert(0, '/home/xut/csclip/sam3')
sys.path.insert(0, '/home/xut/csclip')

results = []

try:
    from biomedclip_zeroshot_cell_classify import InstanceInfo, ensure_local_biomedclip_dir, resolve_device
    results.append("Module 1 OK: biomedclip_zeroshot_cell_classify")
except Exception as e:
    results.append(f"Module 1 FAIL: {e}")

try:
    from biomedclip_fewshot_support_experiment import encode_multiscale_feature, normalize_feature
    results.append("Module 2 OK: biomedclip_fewshot_support_experiment")
except Exception as e:
    results.append(f"Module 2 FAIL: {e}")

try:
    from biomedclip_query_adaptive_classifier import (
        SupportRecord, QueryRecord, SupportCandidate,
        build_morphology_stats, normalize_morphology_feature,
        compute_morphology_features, build_class_morph_prototypes,
    )
    results.append("Module 3 OK: biomedclip_query_adaptive_classifier")
except Exception as e:
    results.append(f"Module 3 FAIL: {e}")

try:
    from biomedclip_hybrid_adaptive_classifier import (
        HybridConfig, SupportReliabilityConfig,
        _compute_query_score_details, _prototypes_from_support_records,
        softmax_np, CLASS_NAMES, LOG_AREA_INDEX, MIN_SIZE_SIGMA,
    )
    results.append("Module 4 OK: biomedclip_hybrid_adaptive_classifier")
except Exception as e:
    results.append(f"Module 4 FAIL: {e}")

try:
    from labeling_tool.fewshot_biomedclip import prepare_classifier, predict_annotations
    results.append("OK: labeling_tool.fewshot_biomedclip")
except Exception as e:
    results.append(f"FAIL fewshot: {e}")

try:
    from labeling_tool.hybrid_classifier import prepare_classifier as hpc
    results.append("OK: labeling_tool.hybrid_classifier")
except Exception as e:
    results.append(f"FAIL hybrid: {e}")

try:
    from labeling_tool.paths import biomedclip_local_dir, sam3_checkpoint_path
    results.append(f"OK: paths (clip={biomedclip_local_dir()}, sam3={sam3_checkpoint_path()})")
except Exception as e:
    results.append(f"FAIL paths: {e}")

try:
    import sam3
    from sam3 import build_sam3_image_model
    results.append(f"OK: sam3 v{sam3.__version__}")
except Exception as e:
    results.append(f"FAIL sam3: {e}")

for r in results:
    print(r)

ok_count = sum(1 for r in results if r.startswith("OK") or r.startswith("Module"))
fail_count = sum(1 for r in results if "FAIL" in r)
print(f"\n{'='*40}")
print(f"Results: {ok_count} OK, {fail_count} FAIL")
