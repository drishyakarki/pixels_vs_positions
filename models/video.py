import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel, AutoConfig


class VideoClassifier(nn.Module):
    """
    video-based classifier for group activity recognition.
    
    supports multiple backbone architectures:
        - image models (dinov3, clip): process frames individually, then aggregate temporally
        - video models (videomae, videomae2): process entire clips with built-in temporal modeling
    
    temporal aggregation options for image models:
        - avgpool: simple average pooling across frames
        - maxpool: max pooling across frames  
        - bilstm: bidirectional LSTM for temporal dependencies
        - tcn: temporal convolutional network for local patterns
        - attention: multi-head self-attention across frames
    """
    
    # mapping of backbone names to HuggingFace model IDs
    BACKBONE_MAP = {
        'dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'clip': 'openai/clip-vit-base-patch16',
        'videomae': 'MCG-NJU/videomae-base',
        'videomae2': 'OpenGVLab/VideoMAEv2-Base'
    }
    
    def __init__(self, backbone_name='dinov3', num_classes=2, 
                 temporal_model='avgpool', num_frames=16, dropout=0.1, 
                 hidden_dim=64, freeze_backbone=True):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.temporal_model = temporal_model
        self.num_frames = num_frames
        self.is_frozen = freeze_backbone
        self.classifier_dropout = dropout
        self.classifier_hidden_dim = hidden_dim
        
        model_id = self.BACKBONE_MAP[backbone_name]
        print(f"Loading {backbone_name}: {model_id}")
        
        # load the appropriate backbone model based on architecture type
        if backbone_name == 'clip':
            self.backbone = CLIPVisionModel.from_pretrained(model_id, use_safetensors=True)
        elif backbone_name == 'videomae2':
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self.backbone = AutoModel.from_pretrained(
                model_id, config=config, trust_remote_code=True, use_safetensors=True
            )
        else:
            self.backbone = AutoModel.from_pretrained(model_id, use_safetensors=True)
        
        # get hidden dimension from backbone config
        if hasattr(self.backbone.config, 'hidden_size'):
            self.hidden_dim = self.backbone.config.hidden_size
        else:
            self.hidden_dim = self.backbone.config.model_config['embed_dim']
        
        # image models process frame-by-frame and need temporal aggregation
        # video models have built-in temporal modeling
        self.is_image_model = backbone_name in ['dinov3', 'clip']
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"{backbone_name} backbone frozen")
        
        # temporal modeling layer (only for image models)
        if self.is_image_model:
            if temporal_model == 'bilstm':
                self.temporal = nn.LSTM(
                    self.hidden_dim, self.hidden_dim, num_layers=2,
                    batch_first=True, bidirectional=True, dropout=0.3
                )
                classifier_input = self.hidden_dim * 2
            elif temporal_model == 'tcn':
                self.temporal = nn.Sequential(
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
                )
                classifier_input = self.hidden_dim
            elif temporal_model == 'attention':
                self.temporal = nn.MultiheadAttention(
                    self.hidden_dim, num_heads=8, dropout=0.1, batch_first=True
                )
                classifier_input = self.hidden_dim
            else:  # avgpool or maxpool
                self.temporal = None
                classifier_input = self.hidden_dim
        else:
            self.temporal = None
            classifier_input = self.hidden_dim
        
        # classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.classifier_dropout),
            nn.Linear(classifier_input, self.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.classifier_dropout * 0.5),
            nn.Linear(self.classifier_hidden_dim, num_classes)
        )
    
    def forward(self, frames):
        """
        forward pass through the video classifier.
        
        args:
            frames: tensor of shape (batch_size, num_frames, height, width, channels)
        
        returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        B, T = frames.shape[0], frames.shape[1]
        
        if self.is_image_model:
            # image models: process each frame independently
            # reshape from (B, T, H, W, C) to (B*T, C, H, W) for batch processing
            frames = frames.permute(0, 1, 4, 2, 3).reshape(B * T, 3, 224, 224)
            
            with torch.set_grad_enabled(self.training and not self.is_frozen):
                outputs = self.backbone(frames)
            
            # extract CLS token from each frame: (B*T, hidden_dim)
            features = outputs.last_hidden_state[:, 0, :]
            # reshape back to (B, T, hidden_dim) to recover temporal structure
            features = features.view(B, T, self.hidden_dim)
            
            # apply temporal aggregation
            if self.temporal_model == 'avgpool':
                features = features.mean(dim=1)
            elif self.temporal_model == 'maxpool':
                features = features.max(dim=1)[0]
            elif self.temporal_model == 'bilstm':
                features, _ = self.temporal(features)
                features = features.mean(dim=1)
            elif self.temporal_model == 'tcn':
                features = features.permute(0, 2, 1)
                features = self.temporal(features)
                features = features.permute(0, 2, 1)
                features = features.mean(dim=1)
            elif self.temporal_model == 'attention':
                features, _ = self.temporal(features, features, features)
                features = features.mean(dim=1)
        else:
            # video models: process entire clip at once
            if self.backbone_name == 'videomae2':
                frames = frames.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
            else:
                frames = frames.permute(0, 1, 4, 2, 3)
            
            with torch.set_grad_enabled(self.training and not self.is_frozen):
                if self.backbone_name == 'videomae2':
                    features = self.backbone.extract_features(pixel_values=frames)
                else:
                    outputs = self.backbone(pixel_values=frames)
                    features = outputs.last_hidden_state[:, 0, :]
        
        return self.classifier(features)