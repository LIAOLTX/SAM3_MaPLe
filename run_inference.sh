python3 inference_maple.py \
    --config configs/sam3_MaPLe_config.yaml \
    --weights outputs/sam3_maple/best_maple_weights.pt \
    --image datasets/COD10K-example/valid/images/COD10K-CAM-3-Flying-64-Moth-4430.jpg \
    --prompt "An inconspicuous moth" \
    --threshold 0.5 \
    --output outputs/output.png