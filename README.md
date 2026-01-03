# WQF7009 Explainable Artificial Intelligence - Assignment 3

Doing some XAI on Chest X-Ray images for Pneumonia classification. Compare different models and XAI techniques.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

Install the dev dependencies:

```bash
uv sync --dev --extra cpu
# If you have a CUDA-enabled GPU, you can install the GPU version:
# uv sync --dev --extra cu130
```

Add extra packages if needed:

```bash
uv add <package-name>
```

## Development

Use jupyter lab/notebook for development:

```bash
# Do this
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
jupyter lab # jupyter notebook

# Or this
uv run jupyter lab  # uv run jupyter notebook
```

Or use any IDE/text editor :)

## Training

To train/eval models, use the `wqf7009-a3-train` or `wqf7009-a3-eval` command with appropriate arguments. For example:

```bash
# SimpleCNN
uv run wqf7009-a3-train --model=simplecnn --epochs=10 --batch-size=32 --data-dir=data/chest_xray --lr=1e-4

# VGG16 (Frozen backbone)
uv run wqf7009-a3-train --model=vgg16 --freeze --epochs=10 --batch-size=32 --data-dir=data/chest_xray --lr=1e-4

# VGG16 (Fine-tuned)
uv run wqf7009-a3-train --model=vgg16 --epochs=10 --batch-size=32 --data-dir=data/chest_xray --lr=1e-4

# ResNet152 (Frozen backbone)
uv run wqf7009-a3-train --model=resnet152 --freeze --epochs=10 --batch-size=32 --data-dir=data/chest_xray --lr=1e-4

# ResNet152 (Fine-tuned)
uv run wqf7009-a3-train --model=resnet152 --epochs=10 --batch-size=32 --data-dir=data/chest_xray --lr=1e-4
```

Some bash scripts that I used (p.s.: I used Google Colab A100 GPU for training, by running the following commands in a Colab's terminal):

```bash
# Setup
# Install pkg
cd "/content/drive/MyDrive/insync/masters/courses/year2526_sem1/wqf7009_explainable_artificial_intelligence/assignment3/wqf7009-a3/"
uv build .
uv pip install dist/wqf7009_a3-0.1.0-py3-none-any.whl

# Copy data to local vm
cp -r /content/drive/MyDrive/insync/masters/courses/year2526_sem1/wqf7009_explainable_artificial_intelligence/assignment3/wqf7009-a3/data/ /content/data

# Train
wqf7009-a3-train --model=simplecnn --epochs=10 --batch-size=128 --data-dir=/content/data/chest_xray --lr=1e-4 --num-workers=4 2>&1 | tee simplecnn.log
wqf7009-a3-train --model=vgg16 --epochs=10 --batch-size=128 --data-dir=/content/data/chest_xray --lr=1e-4 --num-workers=4 2>&1 | tee vgg16_nofreeze.log
wqf7009-a3-train --model=resnet152 --epochs=10 --batch-size=128 --data-dir=/content/data/chest_xray --lr=1e-4 --num-workers=4 2>&1 | tee resnet152_nofreeze.log
wqf7009-a3-train --model=vgg16 --freeze --epochs=10 --batch-size=128 --data-dir=/content/data/chest_xray --lr=1e-4 --num-workers=4 2>&1 | tee vgg16_freeze.log
wqf7009-a3-train --model=resnet152 --freeze --epochs=10 --batch-size=128 --data-dir=/content/data/chest_xray --lr=1e-4 --num-workers=4 2>&1 | tee resnet152_freeze.log

# Evaluate
wqf7009-a3-eval --model=simplecnn \
    --checkpoint=/content/drive/MyDrive/insync/masters/courses/year2526_sem1/wqf7009_explainable_artificial_intelligence/assignment3/wqf7009-a3/models/simplecnn_baseline_10ep_1767264997_best.pth \
    --data-dir=/content/data/chest_xray \
    --batch-size=32 \
    --split=test

wqf7009-a3-eval --model=resnet152 \
    --checkpoint=/content/drive/MyDrive/insync/masters/courses/year2526_sem1/wqf7009_explainable_artificial_intelligence/assignment3/wqf7009-a3/models/resnet152_frozen_10ep_1767266526_best.pth \
    --data-dir=/content/data/chest_xray \
    --batch-size=32 \
    --split=test

wqf7009-a3-eval --model=resnet152 \
    --checkpoint=/content/drive/MyDrive/insync/masters/courses/year2526_sem1/wqf7009_explainable_artificial_intelligence/assignment3/wqf7009-a3/models/resnet152_unfrozen_10ep_1767265644_best.pth \
    --data-dir=/content/data/chest_xray \
    --batch-size=32 \
    --split=test

wqf7009-a3-eval --model=vgg16 \
    --checkpoint=/content/drive/MyDrive/insync/masters/courses/year2526_sem1/wqf7009_explainable_artificial_intelligence/assignment3/wqf7009-a3/models/vgg16_frozen_10ep_1767266050_best.pth \
    --data-dir=/content/data/chest_xray \
    --batch-size=32 \
    --split=test
wqf7009-a3-eval --model=vgg16 \
    --checkpoint=/content/drive/MyDrive/insync/masters/courses/year2526_sem1/wqf7009_explainable_artificial_intelligence/assignment3/wqf7009-a3/models/vgg16_unfrozen_10ep_1767265306_best.pth \
    --data-dir=/content/data/chest_xray \
    --batch-size=32 \
    --split=test
```

## View Training Logs

Training logs are saved in the `runs/` directory. You can use TensorBoard to visualize them:

```bash
uv run tensorboard --logdir=runs/
```

Also, you can checkout the training terminal outputs saved as `.log` files in each model's run directory. For example, `runs/resnet152_frozen_10ep_1767266526/resnet152_freeze.log`

## References

- [Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- [RISE](https://github.com/eclique/RISE/tree/master)
- [Grad-CAM](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html)
