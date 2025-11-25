import torch
import torch.nn as nn
from torch_geometric.nn import (
    DeepGCNLayer, GCNConv, GATv2Conv, SAGEConv, GINConv, 
    EdgeConv, GENConv, GraphConv, TransformerConv, 
    MultiAggregation, global_mean_pool, knn_graph
)


class DeepGCN(nn.Module):
    """
    deep graph convolutional network for tracking-based group activity recognition.
    
    processes sequences of graphs where each graph represents player/ball positions
    at a single timestep. uses DeepGCNLayer with residual connections to enable
    deep architectures (20+ layers).
    
    supports multiple graph convolution types:
        - gcn: standard graph convolution
        - gat: graph attention network (GATv2)
        - sage: GraphSAGE with multi-aggregation
        - gin: graph isomorphism network
        - edgeconv: dynamic edge convolution
        - gen: generalized aggregation
        - graphconv: simple graph convolution
        - transformer: transformer convolution
    
    temporal aggregation options:
        - pool/maxpool: simple pooling across time
        - tcn: temporal convolutional network
        - attention: multi-head self-attention
        - bilstm: bidirectional LSTM
        - transformer_encoder: transformer encoder layers
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=3, num_layers=20, 
                 dropout=0.3, window_size=16, conv_type='gcn', temporal_model='pool'):
        super().__init__()
        
        self.conv_type = conv_type
        self.temporal_model = temporal_model

        if temporal_model == 'bilstm':
            classifier_input = hidden_dim * 2
        else:
            classifier_input = hidden_dim

        # initial linear projection for node features
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        
        # learnable temporal position encoding added after graph pooling
        self.temporal_position_encoding = nn.Parameter(
            torch.randn(1, window_size, hidden_dim) * 0.02
        )
        
        # build GCN layers based on conv_type
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            conv = self._build_conv_layer(conv_type, hidden_dim, dropout)
            norm = nn.LayerNorm(hidden_dim)
            act = nn.ReLU(inplace=True)
            
            layer = DeepGCNLayer(
                conv=conv,
                norm=norm,
                act=act,
                block='res+',
                dropout=dropout
            )
            self.gcn_layers.append(layer)
        
        # build temporal aggregation module
        self.temporal = self._build_temporal_module(temporal_model, hidden_dim, dropout)
        
        # classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def _build_conv_layer(self, conv_type, hidden_dim, dropout):
        """
        builds a graph convolution layer based on the specified type.
        """
        if conv_type == 'gat':
            return GATv2Conv(
                hidden_dim, hidden_dim // 4, heads=4,
                concat=True, dropout=dropout,
                add_self_loops=True, edge_dim=None,
                fill_value='mean', bias=True, share_weights=False
            )
        
        elif conv_type == 'edgeconv':
            return EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                ),
                aggr='max'
            )
        
        elif conv_type == 'sage':
            return SAGEConv(
                hidden_dim, hidden_dim,
                aggr=MultiAggregation(['mean', 'max', 'std']),
                normalize=False, project=True, bias=True
            )
        
        elif conv_type == 'gin':
            return GINConv(
                nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ),
                train_eps=True, eps=0.0
            )
        
        elif conv_type == 'gen':
            return GENConv(
                hidden_dim, hidden_dim,
                aggr='softmax', t=1.0, learn_t=True,
                p=1.0, learn_p=True, msg_norm=True,
                learn_msg_scale=True, norm='layer',
                num_layers=2, eps=1e-7
            )
        
        elif conv_type == 'graphconv':
            return GraphConv(
                hidden_dim, hidden_dim,
                aggr='add', bias=True
            )
        
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
    
    def _build_temporal_module(self, temporal_model, hidden_dim, dropout):
        """
        builds the temporal aggregation module.
        """
        if temporal_model == 'bilstm':
            return nn.LSTM(
                hidden_dim, hidden_dim, num_layers=2,
                batch_first=True, bidirectional=True, dropout=dropout
            )
        
        elif temporal_model == 'tcn':
            return nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
        
        elif temporal_model == 'attention':
            return nn.MultiheadAttention(
                hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )
        
        else:  # pool, maxpool
            return None
    
    def forward(self, batch_data):
        """
        forward pass through the tracking classifier.
        
        args:
            batch_data: dict containing:
                - x: node features (num_nodes, feature_dim)
                - edge_index: edge indices (2, num_edges)
                - batch: batch assignment for each node
                - batch_size: number of samples in batch
                - seq_len: sequence length (window size)
        
        returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        x = self.node_encoder(batch_data['x'])
        
        # message passing through GCN layers
        for layer in self.gcn_layers:
            if self.conv_type == 'edgeconv':
                edge_index = self._compute_edge_dynamic(x, batch_data['batch'])
            else:
                edge_index = batch_data['edge_index']
            x = layer(x, edge_index)
        
        # aggregate node features to graph-level representations
        x = global_mean_pool(x, batch_data['batch'])
        
        # reshape to (batch_size, seq_len, hidden_dim) for temporal processing
        x = x.view(batch_data['batch_size'], batch_data['seq_len'], -1)
        
        # add temporal position encoding
        x = x + self.temporal_position_encoding[:, :batch_data['seq_len'], :]

        # apply temporal aggregation
        if self.temporal_model == 'pool':
            x = torch.mean(x, dim=1)
        
        elif self.temporal_model == 'maxpool':
            x = torch.max(x, dim=1)[0]
        
        elif self.temporal_model == 'tcn':
            x = x.permute(0, 2, 1)
            x = self.temporal(x)
            x = x.permute(0, 2, 1)
            x = torch.max(x, dim=1)[0]
        
        elif self.temporal_model == 'attention':
            x, _ = self.temporal(x, x, x)
            x = torch.max(x, dim=1)[0]
        
        elif self.temporal_model == 'bilstm':
            lstm_out, _ = self.temporal(x)
            x = torch.max(lstm_out, dim=1)[0]
        
        elif self.temporal_model == 'transformer_encoder':
            x = self.temporal(x)
            x = torch.max(x, dim=1)[0]
        
        return self.classifier(x)

    def _compute_edge_dynamic(self, x, batch, k=8):
        """
        computes dynamic edges using knn for EdgeConv.
        """
        k = k or max(8, x.size(0) // batch.max().item() // 3)
        edge_index = knn_graph(x, k=k, batch=batch, loop=False)
        return edge_index
        