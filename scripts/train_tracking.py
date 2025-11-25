import os
import sys
import time
import random
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tracking import PlayerPositionDataset, TeamFlip, HorizontalFlip, VerticalFlip
from models.tracking import DeepGCN


def set_reproducibility(seed=42):
    """
    sets random seeds for reproducibility across runs.
    enables cudnn deterministic mode and sets PYTHONHASHSEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True, warn_only=False)
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    """
    initializes random seeds for dataloader workers.
    ensures reproducibility in multi-worker data loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def custom_collate(batch):
    """
    custom collate function for batching graph sequences.
    combines all graphs from all samples into a single batched graph structure.
    """
    batch_size = len(batch)
    seq_len = batch[0]['seq_len']
    
    all_graphs = []
    for sample_idx, item in enumerate(batch):
        for time_idx, graph in enumerate(item['graphs']):
            graph.sample_idx = sample_idx
            graph.time_idx = time_idx
            all_graphs.append(graph)
    
    batched_graphs = Batch.from_data_list(all_graphs)
    
    return {
        'x': batched_graphs.x,
        'edge_index': batched_graphs.edge_index,
        'batch': batched_graphs.batch,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'label': torch.tensor([item['label'] for item in batch]),
        'window_id': [item['window_id'] for item in batch],
        'label_name': [item['label_name'] for item in batch],
        'game_time': [item['game_time'] for item in batch],
        'team': [item['team'] for item in batch],
        'match_id': [item['match_id'] for item in batch]
    }


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    saves training and validation loss/accuracy curves to file.
    """
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='Train')
    plt.plot(range(1, len(val_accs) + 1), val_accs, 'r-', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plots', 'training_curves.png'), dpi=300)
    plt.close()


def train_model(model, train_loader, val_loader, output_dir, num_epochs, lr, seed, train_dataset, patience=10):
    """
    trains the tracking classifier with gradient clipping and learning rate scheduling.
    implements model reversion to best checkpoint when learning rate is reduced.
    """
    device = torch.device('cuda')
    model = model.to(device)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-8
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 1
    best_val_balanced_acc = 0.0
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    train_balanced_accs, val_balanced_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_correct = train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f"[EPOCH {epoch+1}/{num_epochs}] [TRAIN]")
        for batch in train_pbar:
            batch['x'] = batch['x'].to(device)
            batch['edge_index'] = batch['edge_index'].to(device)
            batch['batch'] = batch['batch'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(batch)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.max(outputs, 1)[1]
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'acc': f"{train_correct/train_total:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_balanced_acc = balanced_accuracy_score(all_train_labels, all_train_preds)
        
        # validation phase
        model.eval()
        val_loss = val_correct = val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        val_pbar = tqdm(val_loader, desc=f"[EPOCH {epoch+1}/{num_epochs}] [VALID]")
        with torch.no_grad():
            for batch in val_pbar:
                batch['x'] = batch['x'].to(device)
                batch['edge_index'] = batch['edge_index'].to(device)
                batch['batch'] = batch['batch'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(batch)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.max(outputs, 1)[1]
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'acc': f"{val_correct/val_total:.4f}"
                })
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_balanced_acc = balanced_accuracy_score(all_val_labels, all_val_preds)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_balanced_accs.append(train_balanced_acc)
        val_balanced_accs.append(val_balanced_acc)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Balanced Acc: {train_balanced_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Balanced Acc: {val_balanced_acc:.4f}")
        print(f"{'='*80}\n")
        
        # save best model based on balanced accuracy
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'seed': seed
            }, os.path.join(output_dir, 'models', 'best_model.pt'))
            print(f"Best model saved at epoch {epoch + 1}")

        # save latest checkpoint for resuming
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'seed': seed,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'train_balanced_accs': train_balanced_accs,
            'val_balanced_accs': val_balanced_accs,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch
        }, os.path.join(output_dir, 'models', 'latest_checkpoint.pt'))
        
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # revert to best model when learning rate is reduced
        if current_lr != prev_lr:
            model.load_state_dict(best_model_state)
            print(f"Learning rate changed from {prev_lr:.2e} to {current_lr:.2e}")
        
        # early stopping when learning rate is too low
        if current_lr <= 2 * scheduler.min_lrs[0]:
            print("\nEarly stopping: learning rate too low")
            break
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    return {
        'best_model_path': os.path.join(output_dir, 'models', 'best_model.pt'),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'acc_at_best_loss': val_accs[best_epoch - 1]
    }


def evaluate_model(model, test_loader, test_dataset, output_dir):
    """
    evaluates the model on the test set and saves confusion matrix and classification report.
    """
    device = torch.device('cuda')
    model = model.to(device).eval()
    
    all_preds = []
    all_labels = []
    
    test_pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch in test_pbar:
            batch['x'] = batch['x'].to(device)
            batch['edge_index'] = batch['edge_index'].to(device)
            batch['batch'] = batch['batch'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(batch)
            preds = torch.max(outputs, 1)[1]
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # sort classes alphabetically for consistent reporting
    class_names = sorted(test_dataset.get_classes())
    class_to_sorted_idx = {test_dataset.label_to_idx[name]: i for i, name in enumerate(class_names)}
    
    sorted_labels = np.array([class_to_sorted_idx[label] for label in all_labels])
    sorted_preds = np.array([class_to_sorted_idx[pred] for pred in all_preds])
    
    cm = confusion_matrix(sorted_labels, sorted_preds)
    per_class_accuracy = np.diag(cm) / cm.sum(axis=1) * 100
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
        f.write(f"{'Class':<30} {'Accuracy':>10}\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<30} {per_class_accuracy[i]:>9.2f}%\n")
        f.write("-" * 60 + "\n\n")
        f.write(classification_report(sorted_labels, sorted_preds, target_names=class_names))
        f.write("-" * 60 + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")
    
    print(f"Balanced Accuracy: {balanced_acc:.2f}%\n")
    print("-" * 60)
    print(f"{'Class':<30} {'Accuracy':>10}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<30} {per_class_accuracy[i]:>9.2f}%")
    print("-" * 60)
    
    return balanced_acc


def main():
    parser = argparse.ArgumentParser(description="Train tracking classifier for group activity recognition")
    parser.add_argument("--data-dir", type=str, default="data/tracking_dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/tracking")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--frame-interval", type=int, default=9)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge", type=str, default="positional", 
                       choices=["ball_knn", "full", "none", "ball_distance", "knn", "distance", "positional"])
    parser.add_argument("--conv-type", type=str, default="gin", 
                       choices=["gcn", "gat", "edgeconv", "sage", "gin", "transformer", "graphconv", "gen"])
    parser.add_argument("--temporal-model", type=str, default="maxpool", 
                       choices=["pool", "tcn", "attention", "bilstm", "maxpool", "transformer_encoder"])
    parser.add_argument("--samples-per-class", type=int, default=4000)
    parser.add_argument("--num-train-games", type=int, default=None)
    parser.add_argument("--num-valid-games", type=int, default=None)

    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"\nEdge type: {args.edge}")
    print(f"Conv type: {args.conv_type}")
    print(f"Output dir: {args.output_dir}")
    print(f"Learning rate: {args.lr}")
    
    set_reproducibility(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # setup augmentations for training
    transforms = None if args.no_augment else [TeamFlip(0.5), HorizontalFlip(0.5), VerticalFlip(0.5)]
    
    # create datasets
    train_dataset = PlayerPositionDataset(
        data_dir=args.data_dir, 
        split='train', 
        window_size=args.window_size, 
        frame_interval=args.frame_interval,
        normalize=True, 
        k=args.k, 
        transforms=transforms, 
        num_workers=args.num_workers, 
        time_tolerance_ms=10, 
        edge=args.edge, 
        random_positioning=True,
        max_position_shift=4,
        random_positioning_prob=0.5,
        seed=args.seed
    )
    
    valid_dataset = PlayerPositionDataset(
        data_dir=args.data_dir, 
        split='valid', 
        window_size=args.window_size, 
        frame_interval=args.frame_interval,
        normalize=True, 
        k=args.k, 
        transforms=None, 
        num_workers=args.num_workers, 
        time_tolerance_ms=10, 
        edge=args.edge, 
        random_positioning=False,
        max_position_shift=4,
        random_positioning_prob=0.5,
        seed=args.seed
    )
    
    test_dataset = PlayerPositionDataset(
        data_dir=args.data_dir, 
        split='test', 
        window_size=args.window_size, 
        frame_interval=args.frame_interval,
        normalize=True, 
        k=args.k, 
        transforms=None, 
        num_workers=args.num_workers, 
        time_tolerance_ms=10, 
        edge=args.edge, 
        random_positioning=False,
        max_position_shift=4,
        random_positioning_prob=0.5,
        seed=args.seed
    )
    
    # optionally reduce training data for scaling experiments
    if args.num_train_games is not None:
        train_dataset._resplit_by_games(args.num_train_games)

    if args.num_valid_games is not None:
        valid_dataset._resplit_by_games(args.num_valid_games)

    if args.num_train_games is not None or args.num_valid_games is not None:
        print("\nAfter resplitting:")
        for dataset, name in [(train_dataset, 'train'), (valid_dataset, 'valid')]:
            label_counts = defaultdict(int)
            for s in dataset.processed_samples:
                label_counts[s['label_name']] += 1
            print(f"\n{name}: {len(dataset.processed_samples)} samples")
            for label in sorted(label_counts.keys()):
                count = label_counts[label]
                percentage = count / len(dataset.processed_samples) * 100
                print(f"    {label:25s}: {count:6d} ({percentage:5.1f}%)")
    
    # print class distribution
    all_class_counts = {}
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        for sample in dataset.processed_samples:
            label = sample['label_name']
            all_class_counts[label] = all_class_counts.get(label, 0) + 1
    
    print("\nOverall class distribution across all splits:")
    total = sum(all_class_counts.values())
    for label in sorted(all_class_counts.keys()):
        count = all_class_counts[label]
        percentage = count / total * 100
        print(f"    {label:25s}: {count:6d} ({percentage:5.1f}%)")
    print('-' * 50)
    
    # setup weighted random sampler for class balancing
    g = torch.Generator()
    g.manual_seed(args.seed)

    class_counts = defaultdict(int)
    for sample in train_dataset.processed_samples:
        class_counts[sample['label']] += 1

    num_classes = len(class_counts)
    samples_per_class = args.samples_per_class
    class_weights = {cls: samples_per_class / count for cls, count in class_counts.items()}

    sample_weights = [class_weights[sample['label']] for sample in train_dataset.processed_samples]  
    sample_weights = torch.DoubleTensor(sample_weights)

    num_samples = args.samples_per_class * num_classes

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
        generator=g
    )
    
    print(f"\nWeightedRandomSampler configured:")
    print(f"  Total samples per epoch: {num_samples}")
    print(f"  Expected samples per class: {args.samples_per_class}")
    print(f"  Number of classes: {num_classes}")
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=custom_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # create model
    model = DeepGCN(
        train_dataset.feature_dim,
        args.hidden_dim, 
        train_dataset.num_classes(),
        args.num_layers,
        args.dropout, 
        args.window_size,
        conv_type=args.conv_type,
        temporal_model=args.temporal_model
    )

    print("\nModel Architecture:")
    print(model)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # train model
    training_results = train_model(
        model, 
        train_loader, 
        val_loader, 
        args.output_dir, 
        args.epochs,
        args.lr, 
        args.seed, 
        train_dataset, 
        args.patience
    )
    
    # evaluate on test set
    checkpoint = torch.load(training_results['best_model_path'], weights_only=True)
    
    model = DeepGCN(
        train_dataset.feature_dim, 
        args.hidden_dim, 
        train_dataset.num_classes(),
        args.num_layers, 
        args.dropout, 
        args.window_size,
        conv_type=args.conv_type,
        temporal_model=args.temporal_model
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc = evaluate_model(model, test_loader, test_dataset, args.output_dir)
    print(f"Balanced Test Accuracy: {test_acc:.2f}%")
    
    end_time = datetime.now()
    total_time = end_time - start_time
    total_seconds = int(total_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTotal Time: {hours:02}:{minutes:02}:{seconds:02}")


if __name__ == "__main__":
    main()