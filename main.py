from src.preprocessing import split_low_quality_images
from src.baseline_evaluation import evaluate_baseline
from src.fine_tuning import fine_tune_model
from src.post_evaluation import evaluate_finetuned_model

if __name__ == "__main__":
    # split_low_quality_images()
    # evaluate_baseline(eval_folder="dataset/eval")
    # fine_tune_model(train_folder="dataset/train")
    evaluate_finetuned_model("dataset/eval")