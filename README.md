# SAM3-MaPLe: Multi-modal Prompt Learning for SAM3

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> **Efficient domain adaptation of Segment Anything Model 3 via multi-modal prompt learning**

We extend [MaPLe (Multi-modal Prompt Learning)](https://arxiv.org/pdf/2210.03117) to adapt SAM3 for downstream segmentation with limited annotated data. By jointly optimizing learnable prompts in both vision and language branches with cross-modal coupling, our method achieves effective domain adaptation while preserving SAM3's generalization capability.

---

## 📌 Highlights

- 🔥 **Parameter-Efficient**: Only **~0.00061%** trainable parameters (**~5K**)
- 🎯 **Multi-modal Alignment**: Joint optimization of vision & language prompts
- 🔄 **Cross-modal Coupling**: Gradient propagation between modalities via projection
- 🧊 **Frozen Backbone**: Pre-trained SAM3 parameters remain fixed
- 📦 **Official Integration**: Built on Meta's official SAM3 training pipeline

---

## 🏗️ Architecture

```
┌─────────────────────┐   ┌─────────────────────┐
│     Input Image     │   │        Text         │
└──────────┬──────────┘   └──────────┬──────────┘
           ▼                         ▼
┌─────────────────────┐   ┌──────────────────────┐
│   Vision Encoder    │   │     Text Encoder     │
│    (Frozen SAM3)    │   │     (Frozen SAM3)    │
│                     │   │   [Prefix][Suffix]   │
│                     │   │     ↑ learnable      │
└──────────┬──────────┘   └──────────┬───────────┘
           │                         │
           │    ┌─────────────────┐  │
           └───►│ Coupling Layer  │◄─┘  ← cross-modal projection
                │   (V ←─── L)    │
                └────────┬────────┘
                         ▼
              ┌─────────────────────┐
              │     Mask Decoder    │
              │    (Frozen SAM3)    │
              └─────────────────────┘
```

**Key Innovation**: Language prompts are projected to vision space through a coupling function, enabling gradient flow between modalities and mutual reinforcement of learned representations.

---

## 📋 Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.12 | 3.12+ |
| PyTorch | 2.7.0 | 2.7.0+ |
| CUDA | 12.6 | 12.6+ |
| GPU VRAM | 16 GB | 24 GB+ |
| Storage | 50 GB | 100 GB+ |

**Knowledge Prerequisites**:
- Familiarity with [Hydra](https://hydra.cc/) configuration system
- Understanding of [COCO dataset format](https://cocodataset.org/#format-data)
- Basic knowledge of [SAM3 architecture](https://arxiv.org/abs/2511.16719)

---

## 🚀 Quick Start

### 1. Install Official SAM3

```bash
# Clone official repository
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# Install with training dependencies
pip install -e ".[train]"
```

### 2. Install SAM3-MaPLe Extension

```bash
# Clone this repository (in a separate directory)
cd /path/to/your/projects
git clone https://github.com/yourusername/sam3-maple.git
cd sam3-maple
```

**Note**: This implementation is built on top of Meta's SAM3 which has its own [license terms](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Please ensure compliance with both licenses when using this code.

---

## 📊 Data Preparation

### Directory Structure

```
your_dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── _annotations.coco.json      # COCO format
└── valid/
    ├── images/
    │   ├── img001.jpg
    │   └── ...
    └── _annotations.coco.json

```

### COCO JSON Format

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img001.jpg",
      "height": 1008,
      "width": 1008
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "moth",
      "supercategory": "insect"
    }
  ]
}
```

### Text Prompt Generation

SAM3-MaPLe uses class names to construct text prompts. Ensure your `categories` names are descriptive. 

```python
# Prompt template: "{prefix} {suffix}"
# Example: "An inconspicuous moth" 
#           ↑prefix          ↑suffix
```

**Note**: Each training session can only train one category. `_annotations.coco.json` will be used. Please handle categories that do not participate in training before training.

---

## ⚙️ Configuration

Create or edit `configs/sam3_MaPLe_config.yaml`:

### Model Configuration

```yaml
model:
  name: "facebook/sam3"
  cache_dir: null
  load_from_HF: false
  checkpoint_path: facebook/sam3/sam3.pt
```

### MaPLe Prompt Learning Configuration

```yaml
MaPLe:
  prefix:
    prompt: An inconspicuous
    tuning: true
  suffix:
    prompt: moth
    tuning: true
```

### Train Configuration

```yaml
# Training settings
training:
  data_dir: datasets/COD10K-example  # Root directory containing train/valid/test folders with COCO annotations
  batch_size: 2      # Increased from 1 to reduce gradient variance
  num_workers: 2        # Increase for better data loading

  learning_rate: 1e-3        # Increased from 1e-5 (SAM3 uses 1e-4 to 5e-4 for fine-tuning)
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  max_grad_norm: 1.0

  num_epochs: 20       # Reduced from 500 (small dataset overfits quickly)
  warmup_steps: 200     # Reduced from 1000 (proportional to dataset size)
  lr_scheduler: "cosine"

  logging_steps: 10
  eval_steps: 1       # More frequent validation (from 500)
  save_steps: 100       # More frequent checkpoints (from 1000)
  save_total_limit: 5   # Keep more checkpoints

  mixed_precision: "bf16"    # Use bfloat16 if available
  seed: 42
  gradient_accumulation_steps: 8  # Reduced from 16 (effective batch: 2×8=16)
```

---

## 🎯 Training

### Basic Training

```bash
python train_sam3_maple.py \
  --config configs/sam3_MaPLe_config.yaml
```

**Expected output**:

```
Building SAM3 model...
Applying MaPLe...
Fine tuning block backbone.language_backbone.prefix_embedding
Fine tuning block backbone.language_backbone.suffix_embedding
Trainable params: 5,120 (0.00061%)

Loading training data from datasets/COD10K-example...
Prefix prompt words: An inconspicuous. Tuning: True
Suffix prompt words: moth. Tuning: True
Loaded COCO dataset: train split
  Images: 49
  Annotations: 49
  Categories: {1: 'moth'}

Loading validation data from datasets/COD10K-example...
Loaded COCO dataset: valid split
  Images: 31
  Annotations: 31
  Categories: {1: 'moth'}
Found validation data: 31 images
Starting training for 20 epochs...
Training samples: 49, Validation samples: 31
Epoch 1:  36%|██████████████                    | 9/25 [00:27<00:45,  2.86s/it, loss=236]
```

**Expected Resource Usage**:

- Trainable parameters: ~5K (~0.00061% of total)
- Training time: ~2-3 hours (1K images, 10 epochs, single RTX 3090TI)
- GPU memory: ~10-12 GB (batch_size=4)

---

## 🔍 Inference

### Single Image Inference

```bash
python3 inference_maple.py \
    --config configs/sam3_MaPLe_config.yaml \
    --weights outputs/sam3_maple/best_maple_weights.pt \
    --image datasets/COD10K-example/valid/images/COD10K-CAM-3-Flying-64-Moth-4430.jpg \
    --prompt "An inconspicuous moth" \
    --threshold 0.5 \
    --output outputs/output.png
```

### Batch Inference

```bash
python inference.py \
  --config configs/sam3_maple.yaml \
  --weights outputs/experiment_1/best_model.pt \
  --input-dir path/to/images/ \
  --class-file classes.txt \
  --output-dir outputs/batch_results/
```

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{liao2026sam3maple,
  author = {Liao, Shu Hao},
  title = {SAM3-MaPLe: Multi-modal Prompt Learning for SAM3},
  howpublished = {https://github.com/yourusername/sam3-maple},
  year = {2026}
}

@inproceedings{khattakMaPLe,
    title={MaPLe: Multi-modal Prompt Learning},
    author={khattak, Muhammad Uzair and Rasheed, Hanoona and Maaz, Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}

@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```

---

## 🙏 Acknowledgments

- [Meta AI](https://ai.meta.com/) for [SAM3](https://github.com/facebookresearch/sam3) and the official training implementation
- [MaPLe authors](https://github.com/muzairkhattak/multimodal-prompt-learning) for the multi-modal prompt learning methodology

---

## 📧 Contact

**Individual Developer**  
📧 Email: ls6311413@gmail.com  

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> **Note**: This implementation is built on top of Meta's SAM3 which has its own [license terms](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Please ensure compliance with both licenses when using this code.