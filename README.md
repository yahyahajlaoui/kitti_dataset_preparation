# 🧠 Deep Learning for Dense Depth Completion  
**Master IMA4201 — Télécom SudParis**

This project introduces **supervised depth completion using deep learning**.

The objective is to predict a **dense depth map** from:
- an RGB image
- a sparse LiDAR depth map
- a validity mask

We use a **UNet-style convolutional neural network** trained on a **small supervised subset of the KITTI Depth Completion dataset**.

---

## 📁 Project Structure

├── kitti.py # Download and prepare KITTI supervised dataset
├── dataloader.py # PyTorch Dataset + DataLoader
├── model.py # UNet architecture
├── train.py # Training loop
├── evaluate.py # Evaluation + visualization
├── utils.py # Metrics and visualization helpers
└── README.md

### ▶️ Command

```bash
# 1) Prepare the KITTI supervised dataset (run once)
python3 kitti.py

# 2) (Optional) Check that the DataLoader works correctly
python3 dataloader.py

# 3) Train the UNet depth completion model
python3 train.py \
  --data_root data/kitti_supervised_3k \
  --epochs 10 \
  --batch_size 4 \
  --lr 1e-4

# 4) Evaluate the trained model and save visualizations
python3 evaluate.py \
  --data_root data/kitti_supervised_3k \
  --ckpt runs/unet_kitti/best.pt
