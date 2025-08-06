"""
Protein Aggregation Prediction Model

A deep learning model that predicts protein aggregation behavior by combining:
1. Protein sequence information using ESM2 transformer embeddings
2. Environmental conditions (temperature, pH, concentration) through neural network processing
3. Multimodal fusion to create unified representations for downstream prediction

Architecture Pipeline:
Protein Sequence → ESM2 → Attention Pooling → [1280 dims]
                                                    ↓
Environmental Data → MLP + BatchNorm → [16 dims]   ↓
                                                    ↓
                          Fusion (Concatenation) → [1296 dims] → MLP → [2 dims] → Prediction

Author: Jeremy H
Date: August 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import EsmTokenizer, EsmModel
from typing import Dict, List, Tuple, Optional, Union


class AttentionPooling(nn.Module):
    """
    Attention pooling layer to reduce sequence embeddings to a fixed size.
    Reduces [batch_size, sequence_length, hidden_dim] -> [batch_size, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [batch_size, sequence_length, hidden_dim]
            attention_mask: [batch_size, sequence_length] - 1 for valid tokens, 0 for padding

        Returns:
            pooled_output: [batch_size, hidden_dim]
            attention_weights: [batch_size, sequence_length]
        """
        # Compute attention scores
        attention_scores = self.attention(
            embeddings
        )  # [batch_size, sequence_length, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, sequence_length]

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(
            attention_scores, dim=1
        )  # [batch_size, sequence_length]

        # Apply attention weights to embeddings
        pooled_output = torch.sum(
            embeddings * attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, hidden_dim]

        return pooled_output, attention_weights


class EnvironmentalEmbedding(nn.Module):
    """
    Environmental embedding layer with batch normalization.
    Converts environmental features to a fixed-dimensional embedding.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 16):
        super(EnvironmentalEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU()

    def forward(self, env_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            env_features: [batch_size, input_dim]

        Returns:
            env_embedding: [batch_size, output_dim]
        """
        x = self.linear(env_features)  # [batch_size, output_dim]
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.activation(x)  # Apply ReLU activation
        return x


class ProteinEnvironmentFusion(nn.Module):
    """
    Fusion module for combining protein sequence embeddings with environmental embeddings.
    Uses concatenation to combine the features.
    """

    def __init__(self, sequence_dim: int = 1280, env_dim: int = 16):
        super(ProteinEnvironmentFusion, self).__init__()
        self.sequence_dim = sequence_dim
        self.env_dim = env_dim
        self.fused_dim = sequence_dim + env_dim

    def forward(
        self, sequence_embedding: torch.Tensor, env_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sequence_embedding: [batch_size, sequence_dim]
            env_embedding: [batch_size, env_dim]

        Returns:
            fused_embedding: [batch_size, sequence_dim + env_dim]
        """
        # Ensure both embeddings have the same batch size
        assert (
            sequence_embedding.shape[0] == env_embedding.shape[0]
        ), f"Batch size mismatch: sequence {sequence_embedding.shape[0]} vs env {env_embedding.shape[0]}"

        # Concatenate along feature dimension
        fused_embedding = torch.cat([sequence_embedding, env_embedding], dim=1)
        return fused_embedding


class AggregationPredictor(nn.Module):
    """
    Multi-layer perceptron for protein aggregation prediction.
    Takes fused embeddings and predicts aggregation outcomes through 4 blocks.
    """

    def __init__(self, input_dim: int = 1296, dropout_rate: float = 0.3):
        super(AggregationPredictor, self).__init__()

        # Block 1: 1296 -> 256
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Block 2: 256 -> 128
        self.block2 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Block 3: 128 -> 64
        self.block3 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Block 4: 64 -> 2 (binary classification: aggregates/doesn't aggregate)
        self.block4 = nn.Sequential(
            nn.Linear(64, 2),
            # Note: No activation here - will apply softmax in training/inference
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_embedding: [batch_size, input_dim]

        Returns:
            logits: [batch_size, 2]
        """
        x = self.block1(fused_embedding)  # [batch_size, 256]
        x = self.block2(x)  # [batch_size, 128]
        x = self.block3(x)  # [batch_size, 64]
        logits = self.block4(x)  # [batch_size, 2]

        return logits


class ProteinAggregationModel(nn.Module):
    """
    Complete protein aggregation prediction model that combines all components.

    This is the main model class that orchestrates:
    1. ESM2 sequence processing with attention pooling
    2. Environmental condition processing
    3. Multimodal fusion
    4. Aggregation prediction
    """

    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        env_input_dim: int = 3,
        env_output_dim: int = 16,
        dropout_rate: float = 0.3,
        freeze_esm: bool = False,
    ):
        super(ProteinAggregationModel, self).__init__()

        # Store configuration
        self.esm_model_name = esm_model_name
        self.env_input_dim = env_input_dim
        self.env_output_dim = env_output_dim
        self.dropout_rate = dropout_rate

        # Load ESM2 model and tokenizer
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)

        # Freeze ESM2 weights if specified
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False

        # Get ESM2 hidden dimension
        self.sequence_dim = self.esm_model.config.hidden_size

        # Initialize all components
        self.attention_pooler = AttentionPooling(self.sequence_dim)
        self.env_embedding = EnvironmentalEmbedding(env_input_dim, env_output_dim)
        self.fusion = ProteinEnvironmentFusion(self.sequence_dim, env_output_dim)
        self.predictor = AggregationPredictor(
            input_dim=self.sequence_dim + env_output_dim, dropout_rate=dropout_rate
        )

    def forward(
        self, protein_sequences: List[str], environmental_conditions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            protein_sequences: List of protein sequences as strings
            environmental_conditions: [batch_size, env_input_dim] tensor with environmental features

        Returns:
            Dictionary containing:
                - logits: [batch_size, 2] aggregation prediction logits
                - probabilities: [batch_size, 2] aggregation prediction probabilities
                - attention_weights: [batch_size, seq_len] attention weights for sequence positions
                - sequence_embedding: [batch_size, sequence_dim] pooled sequence embeddings
                - env_embedding: [batch_size, env_output_dim] environmental embeddings
                - fused_embedding: [batch_size, sequence_dim + env_output_dim] fused embeddings
        """
        batch_size = len(protein_sequences)

        # Tokenize protein sequences
        tokens = self.tokenizer(
            protein_sequences, return_tensors="pt", padding=True, truncation=True
        )

        # Get ESM2 embeddings
        with torch.set_grad_enabled(self.esm_model.training):
            esm_outputs = self.esm_model(**tokens)
            hidden_states = (
                esm_outputs.last_hidden_state
            )  # [batch_size, seq_len, hidden_dim]

        # Apply attention pooling to sequence embeddings
        attention_mask = tokens.get("attention_mask", None)
        sequence_embedding, attention_weights = self.attention_pooler(
            hidden_states, attention_mask
        )

        # Process environmental conditions
        env_embedding = self.env_embedding(environmental_conditions)

        # Fuse embeddings
        fused_embedding = self.fusion(sequence_embedding, env_embedding)

        # Make predictions
        logits = self.predictor(fused_embedding)
        probabilities = F.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": attention_weights,
            "sequence_embedding": sequence_embedding,
            "env_embedding": env_embedding,
            "fused_embedding": fused_embedding,
        }

    def predict(
        self,
        protein_sequences: Union[str, List[str]],
        environmental_conditions: Union[
            Dict[str, float], List[Dict[str, float]], torch.Tensor
        ],
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        """
        High-level prediction interface for easy use.

        Args:
            protein_sequences: Single sequence string or list of sequences
            environmental_conditions: Dict with keys ['Temperature (°C)', 'pH', 'Protein concentration (µM)']
                                    or list of such dicts, or tensor

        Returns:
            Dictionary with prediction results
        """
        self.eval()

        # Handle single sequence input
        if isinstance(protein_sequences, str):
            protein_sequences = [protein_sequences]

        # Handle environmental conditions
        if isinstance(environmental_conditions, dict):
            environmental_conditions = [environmental_conditions]

        if isinstance(environmental_conditions, list):
            # Convert list of dicts to tensor
            env_data = []
            for env_dict in environmental_conditions:
                env_values = [
                    env_dict.get("Temperature (°C)", 37.0),
                    env_dict.get("pH", 7.4),
                    env_dict.get("Protein concentration (µM)", 50.0),
                ]
                env_data.append(env_values)
            environmental_conditions = torch.tensor(env_data, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            outputs = self.forward(protein_sequences, environmental_conditions)

        # Process results
        predictions = []
        for i in range(len(protein_sequences)):
            probs = outputs["probabilities"][i]
            predicted_class = "Aggregates" if probs[1] > 0.5 else "No Aggregation"
            confidence = probs.max().item()

            result = {
                "sequence": protein_sequences[i],
                "prediction": predicted_class,
                "confidence": confidence,
                "prob_no_aggregation": probs[0].item(),
                "prob_aggregation": probs[1].item(),
                "attention_weights": outputs["attention_weights"][i].cpu().numpy(),
            }
            predictions.append(result)

        return predictions if len(predictions) > 1 else predictions[0]

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the model architecture and parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "esm_model": self.esm_model_name,
            "sequence_dim": self.sequence_dim,
            "env_input_dim": self.env_input_dim,
            "env_output_dim": self.env_output_dim,
            "fused_dim": self.sequence_dim + self.env_output_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "dropout_rate": self.dropout_rate,
        }


def create_model(
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    env_input_dim: int = 3,
    env_output_dim: int = 16,
    dropout_rate: float = 0.3,
    freeze_esm: bool = False,
) -> ProteinAggregationModel:
    """
    Factory function to create a protein aggregation prediction model.

    Args:
        esm_model_name: Name of the ESM model to use
        env_input_dim: Number of environmental input features
        env_output_dim: Dimension of environmental embeddings
        dropout_rate: Dropout rate for the prediction MLP
        freeze_esm: Whether to freeze ESM2 weights

    Returns:
        Initialized ProteinAggregationModel
    """
    return ProteinAggregationModel(
        esm_model_name=esm_model_name,
        env_input_dim=env_input_dim,
        env_output_dim=env_output_dim,
        dropout_rate=dropout_rate,
        freeze_esm=freeze_esm,
    )


def analyze_prediction(logits: torch.Tensor) -> List[Dict[str, Union[str, float]]]:
    """
    Utility function to analyze prediction logits and return interpretable results.

    Args:
        logits: [batch_size, 2] prediction logits

    Returns:
        List of dictionaries with prediction analysis
    """
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    confidence = torch.max(probs, dim=1)[0]

    results = []
    for i in range(logits.shape[0]):
        result = {
            "prediction": "Aggregates" if predictions[i] == 1 else "No Aggregation",
            "confidence": confidence[i].item(),
            "prob_no_aggregation": probs[i][0].item(),
            "prob_aggregation": probs[i][1].item(),
            "logit_no_aggregation": logits[i][0].item(),
            "logit_aggregation": logits[i][1].item(),
        }
        results.append(result)

    return results


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model()

    # Example prediction
    sequence = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV"
    env_conditions = {
        "Temperature (°C)": 37.0,
        "pH": 7.4,
        "Protein concentration (µM)": 50.0,
    }

    # Make prediction
    result = model.predict(sequence, env_conditions)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")

    # Model info
    info = model.get_model_info()
    print(f"Model has {info['trainable_parameters']:,} trainable parameters")
