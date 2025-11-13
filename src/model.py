import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualNetWithEmbeddings(nn.Module):
    """
    MLP-based residual predictor with city and location embeddings.
    Includes BatchNorm, adaptive dropout, and learnable bias correction.
    """

    def __init__(self, num_numeric_features, n_locations, n_cities,
                 hidden=128, emb_dim_loc=8, emb_dim_city=4,
                 dropout=0.2, adaptive_dropout=True):
        super().__init__()

        # --- Embeddings ---
        self.loc_emb = nn.Embedding(n_locations, emb_dim_loc)
        self.city_emb = nn.Embedding(n_cities, emb_dim_city)

        # --- Config flags ---
        self.adaptive_dropout = adaptive_dropout
        self.dropout_init = dropout

        # --- Input dimension ---
        input_dim = num_numeric_features + emb_dim_loc + emb_dim_city

        # --- Layers ---
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc_out = nn.Linear(hidden, 1)

        # --- Dropout layers (can be updated dynamically) ---
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

        # --- Learnable bias correction parameter ---
        self.bias_correction = nn.Parameter(torch.zeros(1))

        # Optional: Xavier initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_num, loc_id, city_id):
        # --- Embedding lookups ---
        loc_emb = self.loc_emb(loc_id)
        city_emb = self.city_emb(city_id)

        # --- Combine numeric + embedding features ---
        x = torch.cat([x_num, loc_emb, city_emb], dim=1)

        # --- Forward through MLP ---
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        # --- Output + bias correction ---
        out = self.fc_out(x) + self.bias_correction
        return out

    def set_dropout(self, p: float):
        """Utility to update dropout probability during training."""
        self.drop1.p = p
        self.drop2.p = p
