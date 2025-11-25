import os
import sys
import random
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tracking import PlayerPositionDataset
from models.tracking import DeepGCN


def set_reproducibility(seed=42):
    """
    sets random seeds for reproducibility across runs.
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
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def custom_collate(batch):
    """
    custom collate function for batching graph sequences.
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


def evaluate_model(model, test_loader, test_dataset, output_dir):
    """
    evaluates the model on the test set and saves confusion matrix and classification report.
    """
    device = torch.device('cuda')
    model = model.to(device).eval()
    
    all_preds = []
    all_labels = []
    all_metadata = []
    
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

            for i in range(len(batch['window_id'])):
                all_metadata.append({
                    'match_id': batch['match_id'][i],
                    'game_time': batch['game_time'][i],
                    'team': batch['team'][i],
                    'label': batch['label_name'][i],
                    'predicted': test_dataset.idx_to_label[preds[i].item()]
                })
    
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

    with open(os.path.join(output_dir, 'results', 'predictions.json'), 'w') as f:
        json.dump({'predictions': all_metadata}, f, indent=2)
    
    return balanced_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking classifier for group activity recognition")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/tracking_dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/tracking/eval")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--frame-interval", type=int, default=9)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge", type=str, default="positional", 
                       choices=["ball_knn", "full", "none", "ball_distance", "knn", "distance", "positional"])
    parser.add_argument("--conv-type", type=str, default="gin", 
                       choices=["gcn", "gat", "edgeconv", "sage", "gin", "transformer", "graphconv", "gen"])
    parser.add_argument("--temporal-model", type=str, default="attention", 
                       choices=["pool", "tcn", "attention", "bilstm", "maxpool", "transformer_encoder"])
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    
    set_reproducibility(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("TRACKING EVALUATION")
    print("-"*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print("-"*60 + "\n")
    
    # create dataset
    test_dataset = PlayerPositionDataset(
        data_dir=args.data_dir, 
        split='test', 
        window_size=16, 
        frame_interval=9,
        normalize=True, 
        k=8, 
        transforms=None, 
        num_workers=args.num_workers, 
        time_tolerance_ms=10, 
        edge=args.edge, 
        random_positioning=False,
        max_position_shift=4,
        random_positioning_prob=0.5,
        seed=args.seed
    )
    
    print(f"\nTest: {len(test_dataset)} samples")
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
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
    
    # create model and load checkpoint
    model = DeepGCN(
        test_dataset.feature_dim, 
        args.hidden_dim, 
        test_dataset.num_classes(),
        args.num_layers, 
        args.dropout, 
        args.window_size,
        conv_type=args.conv_type,
        temporal_model=args.temporal_model
    )
    
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nLoaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # evaluate
    print("\nEvaluating...")
    balanced_acc = evaluate_model(model, test_loader, test_dataset, args.output_dir)
    print(f"\nFinal Balanced Accuracy: {balanced_acc:.2f}%")


if __name__ == "__main__":
    main()