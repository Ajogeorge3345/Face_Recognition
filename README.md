# Fine-Tuning Face Recognition on Low-Quality Images

This project demonstrates how to fine-tune a face recognition model to work better with low-quality images, such as those from CCTV cameras or poor lighting environments. It includes dataset preprocessing, baseline evaluation, contrastive learning-based fine-tuning, and post-evaluation analysis.

## 🚀 Project Overview

The pipeline consists of several stages:
1. **Dataset Preprocessing** - Split low-quality images into train/eval sets
2. **Baseline Evaluation** - Evaluate original InsightFace model performance
3. **Fine-Tuning** - Train a neural network adapter using contrastive learning
4. **Post-Evaluation** - Compare fine-tuned model performance with baseline

## 📂 Project Structure

```
Face_Recognition/
├── Images/                          # Raw dataset
│   ├── person1/
│   │   ├── high_quality/
│   │   └── low_quality/
│   ├── person2/
│   │   ├── high_quality/
│   │   └── low_quality/
│   └── ...
├── dataset/                         # Processed dataset (created after preprocessing)
│   ├── train/
│   │   ├── person1/
│   │   ├── person2/
│   │   └── ...
│   └── eval/
│       ├── person1/
│       ├── person2/
│       └── ...
├── outputs/                         # Generated results
│   ├── fine_tuned_model.pt
│   ├── roc_curve.png
│   ├── precision_recall.png
│   └── similarity_distribution.png
├── src/
│   ├── preprocessing.py
│   ├── baseline_evaluation.py
│   ├── fine_tuning.py
│   └── post_evaluation.py
├── main.py                          # Main execution script
├── requirements.txt
└── README.md
```

## 💻 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Ajogeorge3345/Face_Recognition.git
cd Face_Recognition
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Your Dataset
Ensure your dataset folder structure looks like this:
```
Images/
├── person1/
│   ├── high_quality/     # High quality images (not used in training)
│   └── low_quality/      # Low quality images (used for training/evaluation)
├── person2/
│   ├── high_quality/
│   └── low_quality/
└── ...
```

**Note**: Only the `low_quality` folders are used for training and evaluation. The `high_quality` folders are optional and not processed by the current pipeline.

## 🔄 Step-by-Step Execution

### Step 1: Data Preprocessing
Split low-quality images into training and evaluation sets (70-30 split):

```bash
python -c "from src.preprocessing import split_low_quality_images; split_low_quality_images()"
```

Or edit `main.py` to uncomment the preprocessing line and run:
```bash
python main.py
```

This creates the `dataset/` folder with `train/` and `eval/` subdirectories.

### Step 2: Baseline Evaluation
Evaluate the original InsightFace model performance:

```bash
python -c "from src.baseline_evaluation import evaluate_baseline; evaluate_baseline('dataset/eval')"
```

This generates baseline performance plots in the `outputs/` folder.

### Step 3: Fine-Tune the Model
Train the fine-tuning adapter using contrastive learning:

```bash
python -c "from src.fine_tuning import fine_tune_model; fine_tune_model('dataset/train')"
```

This saves the fine-tuned model to `outputs/fine_tuned_model.pt`.

### Step 4: Post-Fine-Tuning Evaluation
Evaluate the fine-tuned model performance:

```bash
python -c "from src.post_evaluation import evaluate_finetuned_model; evaluate_finetuned_model('dataset/eval')"
```

This generates fine-tuned performance plots with "_finetuned" suffix in the `outputs/` folder.

### Alternative: Run Complete Pipeline
You can also run the complete pipeline by editing `main.py` and uncommenting the desired steps:

```python
if __name__ == "__main__":
    split_low_quality_images()                    # Step 1
    evaluate_baseline(eval_folder="dataset/eval") # Step 2  
    fine_tune_model(train_folder="dataset/train") # Step 3
    evaluate_finetuned_model("dataset/eval")      # Step 4
```

Then run:
```bash
python main.py
```

## 📊 Output Files

The project generates several evaluation metrics and visualizations:

### Baseline Results:
- `similarity_distribution.png` - Distribution of similarity scores
- `roc_curve.png` - ROC curve and AUC score
- `precision_recall.png` - Precision-recall curve

### Fine-tuned Results:
- `similarity_distribution_finetuned.png` - Fine-tuned similarity distribution
- `roc_curve_finetuned.png` - Fine-tuned ROC curve and AUC score
- `precision_recall_finetuned.png` - Fine-tuned precision-recall curve
- `fine_tuned_model.pt` - Saved fine-tuned model weights

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: InsightFace (buffalo_l) - pre-trained face recognition model
- **Fine-tuning Network**: Simple 3-layer MLP (512 → 256 → 128 dimensions)
- **Loss Function**: Contrastive loss with margin-based distance learning

### Training Details
- **Epochs**: 10 (configurable)
- **Batch Size**: 64 (configurable)
- **Learning Rate**: 1e-3 (configurable)
- **Optimizer**: Adam
- **Device**: Automatically detects CUDA/CPU

### Data Split
- **Training**: 70% of low-quality images per person
- **Evaluation**: 30% of low-quality images per person
- **Pair Generation**: 
  - Positive pairs: Same person images
  - Negative pairs: Different person images

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA Provider Warning**: 
   - This is normal if you don't have CUDA-enabled GPU
   - The model will automatically fall back to CPU execution

2. **Missing Model File Error**:
   - Ensure you've run the fine-tuning step before post-evaluation
   - Check that `outputs/fine_tuned_model.pt` exists

3. **No Face Detected**:
   - Ensure images contain clear, detectable faces
   - Check image quality and lighting conditions

4. **Import Errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure you're in the project root directory

5. **Module Import Issues**:
   - Make sure you're running commands from the project root directory
   - If using `main.py`, ensure the correct functions are uncommented

## 📋 Requirements

- Python 3.7+
- PyTorch
- InsightFace
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- ONNX Runtime
- tqdm
- albumentations

## 📈 Expected Results

The fine-tuning process should improve face recognition performance on low-quality images by:
- Better separation between same/different person similarities
- Higher AUC scores in ROC curves
- Improved precision-recall metrics
- More robust embeddings for poor quality images

## ⚠️ Important Notes

1. **Sequential Execution**: Run steps in order - preprocessing must be done before evaluation, fine-tuning before post-evaluation
2. **Dataset Requirements**: Ensure you have sufficient images per person for meaningful training and evaluation
3. **Computational Requirements**: Fine-tuning may take time depending on dataset size and available hardware
4. **Model Loading**: The post-evaluation step requires the fine-tuned model to exist

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!
