import os
import random
import argparse
from datetime import datetime
import sys


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.video import VideoDataset
from models.video import VideoClassifier


def set_reproducibility(seed=42):
    """
    sets random seeds for reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def seed_worker(worker_id):
    """
    initializes random seeds for dataloader workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def evaluate_model(model, test_loader, test_dataset, output_dir):
    """
    evaluates the model on the test set and saves confusion matrix and classification report.
    """
    device = torch.device('cuda')
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.eval()
    
    class_names = test_dataset.get_classes()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device, non_blocking=True, dtype=torch.float32)
            
            with torch.amp.autocast('cuda'):
                outputs = model(frames)
            
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    class_names = sorted(test_dataset.get_classes())
    class_to_sorted_idx = {test_dataset.label_to_idx[name]: i for i, name in enumerate(class_names)}

    sorted_labels = np.array([class_to_sorted_idx[label] for label in all_labels])
    sorted_preds = np.array([class_to_sorted_idx[pred] for pred in all_preds])

    cm = confusion_matrix(sorted_labels, sorted_preds)
    per_class_accuracy = np.diag(cm) / (cm.sum(axis=1)) * 100
    balanced_acc = balanced_accuracy_score(sorted_labels, sorted_preds) * 100
    
    # save confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # save classification report
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    with open(os.path.join(output_dir, 'results', 'classification_report.txt'), 'w') as f:
        f.write(f"Balanced Accuracy: {balanced_acc:.2f}%\n\n")
        f.write(f"{'Class':<30} {'Accuracy':>10} {'Samples':>10}\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            num_samples = cm[i].sum()
            acc_str = f"{per_class_accuracy[i]:>9.2f}%" if num_samples > 0 else "     N/A"
            f.write(f"{class_name:<30} {acc_str} {int(num_samples):>10}\n")
        f.write("-" * 60 + "\n\n")
        f.write(classification_report(
            sorted_labels, sorted_preds, target_names=class_names,
            zero_division=0
        ))
    
    print(f"Balanced Accuracy: {balanced_acc:.2f}%\n")
    print("-" * 60)
    print(f"{'Class':<30} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        num_samples = cm[i].sum()
        acc_str = f"{per_class_accuracy[i]:>9.2f}%" if num_samples > 0 else "     N/A"
        print(f"{class_name:<30} {acc_str} {int(num_samples):>10}")
    print("-" * 60)
    
    return balanced_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate video classifier for group activity recognition")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--preprocessed-dir", type=str, default="data/video_dataset/preprocessed")
    parser.add_argument("--output-dir", type=str, default="outputs/video/eval")
    parser.add_argument("--backbone", type=str, default="dinov3",
                       choices=["dinov3", "clip", "videomae", "videomae2", "timesformer"])
    parser.add_argument("--temporal-model", type=str, default="avgpool",
                       choices=["avgpool", "maxpool", "bilstm", "tcn", "attention"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    start_time = datetime.now()
    
    set_reproducibility(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("VIDEO EVALUATION")
    print("-"*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backbone: {args.backbone}")
    print(f"Output: {args.output_dir}")
    print("-"*60 + "\n")
    
    # create dataset
    test_dataset = VideoDataset(
        preprocessed_dir=args.preprocessed_dir, 
        split='test', 
        augment=False
    )
    
    print(f"\nTest: {len(test_dataset)} clips")
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    def collate_fn(batch):
        frames = torch.FloatTensor(np.array([item[0] for item in batch]))
        labels = torch.LongTensor([item[1] for item in batch])
        return frames, labels
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker, 
        generator=g,
        persistent_workers=True, 
        prefetch_factor=4, 
        collate_fn=collate_fn
    )
    
    # create model and load checkpoint
    model = VideoClassifier(
        backbone_name=args.backbone,
        num_classes=test_dataset.num_classes(),
        temporal_model=args.temporal_model,
        freeze_backbone=args.freeze_backbone,
        num_frames=16
    )
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nLoaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # evaluate
    print("\nEvaluating...")
    balanced_acc = evaluate_model(model, test_loader, test_dataset, args.output_dir)
    print(f"\nFinal Balanced Accuracy: {balanced_acc:.2f}%")

    end_time = datetime.now()
    total_time = end_time - start_time
    total_seconds = int(total_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTotal Time: {hours:02}:{minutes:02}:{seconds:02}")

if __name__ == "__main__":
    main()