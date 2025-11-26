import os
import random
import argparse
from datetime import datetime
from collections import defaultdict
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.video import VideoDataset, preprocess_split
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
    ensures reproducibility in multi-worker data loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def mixup_data(x, y, alpha=0.2):
    """
    applies mixup augmentation by blending pairs of samples and their labels.
    helps improve generalization by creating virtual training examples.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    computes mixed loss for mixup-augmented samples.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    saves training and validation loss/accuracy curves to file.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Valid')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Loss Curves')
    
    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Valid')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Accuracy Curves')
    
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plots', 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def train_model(model, train_loader, val_loader, output_dir, args, train_dataset, start_epoch=0, history=None):
    """
    trains the video classifier with mixed precision, gradient clipping, and learning rate scheduling.
    implements model reversion to best checkpoint when learning rate is reduced.
    """
    device = torch.device('cuda')
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=args.patience, min_lr=1e-8
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_model_state = None
    
    # restore training state if resuming
    if history is not None and 'optimizer_state_dict' in history:
        optimizer.load_state_dict(history['optimizer_state_dict'])
        scheduler.load_state_dict(history['scheduler_state_dict'])
        if 'scaler_state_dict' in history:
            scaler.load_state_dict(history['scaler_state_dict'])
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        train_accs = history['train_accs']
        val_accs = history['val_accs']
        train_balanced_accs = history.get('train_balanced_accs', [])
        val_balanced_accs = history.get('val_balanced_accs', [])
        best_val_balanced_acc = history['best_val_balanced_acc']
        best_epoch = history['best_epoch']
        
        best_model_path = os.path.join(output_dir, 'models', 'best_model.pt')
        if os.path.exists(best_model_path):
            best_checkpoint = torch.load(best_model_path)
            best_model_state = best_checkpoint['model_state_dict']
    else:
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        train_balanced_accs, val_balanced_accs = [], []
        best_val_balanced_acc = 0.0
        best_epoch = 0
    
    print(f"\nStarting training from epoch {start_epoch}...")
    print("-" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_preds_all = []
        train_labels_all = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(device, non_blocking=True, dtype=torch.float32)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # apply mixup with 50% probability when augmentation is enabled
            use_mixup = train_dataset.augment and np.random.random() > 0.5
            
            with torch.amp.autocast('cuda'):
                if use_mixup:
                    frames, labels_a, labels_b, lam = mixup_data(frames, labels, alpha=args.mixup_alpha)
                    outputs = model(frames)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            
            if use_mixup:
                correct = (lam * predicted.eq(labels_a).sum().item() + 
                          (1 - lam) * predicted.eq(labels_b).sum().item())
                train_correct += correct
                train_preds_all.extend(predicted.cpu().numpy())
                train_labels_all.extend(labels_a.cpu().numpy())
            else:
                train_correct += (predicted == labels).sum().item()
                train_preds_all.extend(predicted.cpu().numpy())
                train_labels_all.extend(labels.cpu().numpy())
            
            train_total += labels.size(0)
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'acc': f'{train_correct/train_total:.3f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_balanced_acc = balanced_accuracy_score(train_labels_all, train_preds_all)
        
        # validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        val_preds_all = []
        val_labels_all = []
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                frames = frames.to(device, non_blocking=True, dtype=torch.float32)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_balanced_acc = balanced_accuracy_score(val_labels_all, val_preds_all)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_balanced_accs.append(train_balanced_acc)
        val_balanced_accs.append(val_balanced_acc)
        
        print(f"\n{'-'*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Balanced Acc: {train_balanced_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Balanced Acc: {val_balanced_acc:.4f}")
        print(f"{'-'*60}\n")
        
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # save best model based on balanced accuracy
        if val_balanced_acc > best_val_balanced_acc:
            best_model_state = model_to_save.state_dict()
            best_val_balanced_acc = val_balanced_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'seed': args.seed
            }, os.path.join(output_dir, 'models', 'best_model.pt'))
            print(f"\nBest model saved (Val Balanced Acc: {best_val_balanced_acc:.4f})")
        
        # save latest checkpoint for resuming
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'seed': args.seed,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'train_balanced_accs': train_balanced_accs,
            'val_balanced_accs': val_balanced_accs,
            'best_val_balanced_acc': best_val_balanced_acc,
            'best_epoch': best_epoch
        }, os.path.join(output_dir, 'models', 'latest_checkpoint.pt'))
        
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # revert to best model when learning rate is reduced
        if current_lr != prev_lr:
            print(f"\nLearning rate reduced: {prev_lr:.2e} -> {current_lr:.2e}")
            if best_model_state is not None:
                print(f"\nReverting to best model state from epoch {best_epoch}")
                model_to_load = model.module if hasattr(model, 'module') else model
                model_to_load.load_state_dict(best_model_state)
            else:
                print(f"\nWarning: No best model state available to revert to")

        # early stopping when learning rate is too low
        if current_lr <= 2 * scheduler.min_lrs[0]:
            print("\nEarly stopping: learning rate too low")
            break
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    print(f"\nTraining complete. Best model: Epoch {best_epoch}, Val Balanced Acc: {best_val_balanced_acc:.4f}")
    
    return os.path.join(output_dir, 'models', 'best_model.pt')


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
    parser = argparse.ArgumentParser(description="Train video classifier for group activity recognition")
    parser.add_argument("--data-dir", type=str, default="data/video_dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/video")
    parser.add_argument("--preprocessed-dir", type=str, default="data/video_dataset/preprocessed")
    parser.add_argument("--backbone", type=str, default="videomae2",
                       choices=["dinov3", "clip", "videomae", "videomae2", "timesformer"])
    parser.add_argument("--temporal-model", type=str, default="maxpool",
                       choices=["avgpool", "maxpool", "bilstm", "tcn", "attention"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action='store_true')
    parser.add_argument("--preprocess", action='store_true')
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--samples-per-class", type=int, default=4000)
    parser.add_argument("--num-train-games", type=int, default=None)
    parser.add_argument("--num-valid-games", type=int, default=None)

    args = parser.parse_args()

    start_time = datetime.now()    
    set_reproducibility(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("VIDEO CLASSIFICATION")
    print("="*60)
    print(f"Backbone: {args.backbone}")
    if args.backbone in ['dinov3', 'clip']:
        print(f"Temporal Model: {args.temporal_model}")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")
    
    # preprocess splits if needed
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(args.preprocessed_dir, split)
        if args.preprocess or not os.path.exists(os.path.join(split_dir, 'clips.json')):
            print(f"\nPreprocessing {split}...")
            preprocess_split(args.data_dir, split, split_dir)
    
    # create datasets
    train_dataset = VideoDataset(
        preprocessed_dir=args.preprocessed_dir, 
        split='train', 
        augment=not args.no_augment
    )
    valid_dataset = VideoDataset(
        preprocessed_dir=args.preprocessed_dir, 
        split='valid', 
        augment=False
    )
    test_dataset = VideoDataset(
        preprocessed_dir=args.preprocessed_dir, 
        split='test', 
        augment=False
    )

    # optionally reduce training data for scaling experiments
    if args.num_train_games is not None:
        train_dataset._resplit_by_games(args.num_train_games, args.data_dir)

    if args.num_valid_games is not None:
        valid_dataset._resplit_by_games(args.num_valid_games, args.data_dir)

    if args.num_train_games is not None or args.num_valid_games is not None:
        print("\nAfter resplitting by games:")
        for dataset, name in [(train_dataset, 'train'), (valid_dataset, 'valid')]:
            label_counts = defaultdict(int)
            for c in dataset.clips:
                label_counts[c['label']] += 1
            
            print(f"\n{name.upper()}: {len(dataset.clips)} clips")
            for label in sorted(label_counts.keys()):
                count = label_counts[label]
                percentage = count / len(dataset.clips) * 100
                print(f"  {label:25s}: {count:6d} ({percentage:5.1f}%)")
    
    print("\n", "-"*60, "\n")
    print(f"Train: {len(train_dataset)}")
    print(f"Valid: {len(valid_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print(f"Total: {len(train_dataset) + len(valid_dataset) + len(test_dataset)}")

    # print class distribution
    all_class_counts = {}
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        for clip in dataset.clips:
            label_name = clip['label']
            all_class_counts[label_name] = all_class_counts.get(label_name, 0) + 1

    print("\nOverall class distribution:")
    total = sum(all_class_counts.values())
    for label in sorted(all_class_counts.keys()):
        count = all_class_counts[label]
        percentage = count / total * 100
        print(f"    {label:25s}: {count:6d} ({percentage:5.1f}%)")
    print('-' * 60)

    # setup weighted random sampler for class balancing
    g = torch.Generator()
    g.manual_seed(args.seed)

    class_counts = defaultdict(int)
    for clip in train_dataset.clips:
        class_counts[clip['label_idx']] += 1

    num_classes = len(class_counts)
    samples_per_class = args.samples_per_class
    class_weights = {cls: samples_per_class / count for cls, count in class_counts.items()}
    
    sample_weights = [class_weights[clip['label_idx']] for clip in train_dataset.clips]
    sample_weights = torch.DoubleTensor(sample_weights)

    num_samples = args.samples_per_class * num_classes

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )

    print(f"\nWeightedRandomSampler configured:")
    print(f"  Total samples per epoch: {num_samples}")
    print(f"  Expected samples per class: {args.samples_per_class}")
    print(f"  Number of classes: {num_classes}")
    
    def collate_fn(batch):
        frames = torch.FloatTensor(np.array([item[0] for item in batch]))
        labels = torch.LongTensor([item[1] for item in batch])
        return frames, labels
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=seed_worker, 
        generator=g,
        persistent_workers=True, 
        prefetch_factor=4, 
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        valid_dataset, 
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
    
    # create model
    model = VideoClassifier(
        backbone_name=args.backbone,
        num_classes=train_dataset.num_classes(),
        temporal_model=args.temporal_model,
        freeze_backbone=args.freeze_backbone,
        num_frames=16
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # handle training resumption
    start_epoch = 0
    history = None
    if args.resume:
        print(f"\nResuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        history = {
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'train_accs': checkpoint.get('train_accs', []),
            'val_accs': checkpoint.get('val_accs', []),
            'train_balanced_accs': checkpoint.get('train_balanced_accs', []),
            'val_balanced_accs': checkpoint.get('val_balanced_accs', []),
            'best_val_balanced_acc': checkpoint.get('best_val_balanced_acc', 0.0),
            'best_epoch': checkpoint.get('best_epoch', 0),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
            'scaler_state_dict': checkpoint.get('scaler_state_dict')
        }

    # train model
    print("\nTraining...")
    best_model_path = train_model(
        model, 
        train_loader, 
        val_loader, 
        args.output_dir, 
        args, 
        train_dataset,
        start_epoch,
        history
    )
    
    # evaluate on test set
    print("\nEvaluating...")
    checkpoint = torch.load(best_model_path)
    
    model = VideoClassifier(
        backbone_name=args.backbone,
        num_classes=test_dataset.num_classes(),
        temporal_model=args.temporal_model,
        freeze_backbone=args.freeze_backbone,
        num_frames=16
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    balanced_acc = evaluate_model(model, test_loader, test_dataset, args.output_dir)
    print(f"Balanced Accuracy: {balanced_acc:.2f}%")
    
    end_time = datetime.now()
    total_time = end_time - start_time
    total_seconds = int(total_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTotal Time: {hours:02}:{minutes:02}:{seconds:02}")

if __name__ == "__main__":
    main()