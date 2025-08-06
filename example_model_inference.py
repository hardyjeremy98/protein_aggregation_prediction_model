"""
Example usage of the Protein Aggregation Prediction Model

This script demonstrates how to use the model.py file for protein aggregation prediction.
"""

import torch
import numpy as np
from model import create_model, ProteinAggregationModel


def main():
    print("🧬 Protein Aggregation Prediction Model - Example Usage\n")

    # Create the model
    print("1. Creating model...")
    model = create_model(
        esm_model_name="facebook/esm2_t33_650M_UR50D",
        env_input_dim=3,
        env_output_dim=16,
        dropout_rate=0.3,
        freeze_esm=False,  # Set to True to freeze ESM2 weights during training
    )

    # Display model information
    info = model.get_model_info()
    print(f"   Model loaded: {info['esm_model']}")
    print(f"   Total parameters: {info['trainable_parameters']:,}")
    print(
        f"   Architecture: Sequence({info['sequence_dim']}) + Env({info['env_output_dim']}) -> Fused({info['fused_dim']}) -> Prediction(2)"
    )

    # Example 1: Single protein prediction
    print("\n2. Single protein prediction...")
    sequence = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV"
    env_conditions = {
        "Temperature (°C)": 37.0,
        "pH": 7.4,
        "Protein concentration (µM)": 50.0,
    }

    result = model.predict(sequence, env_conditions)
    print(f"   Sequence: {sequence}")
    print(
        f"   Environment: T={env_conditions['Temperature (°C)']}°C, pH={env_conditions['pH']}, C={env_conditions['Protein concentration (µM)']}µM"
    )
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   P(No Aggregation): {result['prob_no_aggregation']:.4f}")
    print(f"   P(Aggregation): {result['prob_aggregation']:.4f}")

    # Example 2: Batch prediction
    print("\n3. Batch prediction...")
    sequences = [
        "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV",
        "MKFLVLLFNISCSFAGTAITQTQTTQTTQTTQTTQTTQTT",
        "GGSGGSEQAFAAGGSLALAASGGGGKRSLVTDKCSASLAL",
    ]

    env_batch = [
        {"Temperature (°C)": 25.0, "pH": 6.8, "Protein concentration (µM)": 100.0},
        {"Temperature (°C)": 37.0, "pH": 7.4, "Protein concentration (µM)": 50.0},
        {"Temperature (°C)": 42.0, "pH": 8.0, "Protein concentration (µM)": 25.0},
    ]

    batch_results = model.predict(sequences, env_batch)

    for i, result in enumerate(batch_results):
        print(f"   Sample {i+1}:")
        print(f"     Sequence: {result['sequence'][:30]}...")
        print(
            f"     Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})"
        )

    # Example 3: Using the raw forward pass
    print("\n4. Raw forward pass (for training/advanced usage)...")

    # Prepare environmental conditions as tensor
    env_tensor = torch.tensor(
        [[37.0, 7.4, 50.0], [25.0, 6.8, 100.0]],  # Temperature, pH, Concentration
        dtype=torch.float32,
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model.forward(sequences[:2], env_tensor)

    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Probabilities shape: {outputs['probabilities'].shape}")
    print(f"   Attention weights shape: {outputs['attention_weights'].shape}")
    print(f"   Sequence embedding shape: {outputs['sequence_embedding'].shape}")
    print(f"   Environmental embedding shape: {outputs['env_embedding'].shape}")
    print(f"   Fused embedding shape: {outputs['fused_embedding'].shape}")

    # Example 4: Attention analysis
    print("\n5. Attention analysis...")
    attention_weights = result["attention_weights"]
    sequence_chars = list(sequence)

    # Find top 5 positions with highest attention
    top_indices = np.argsort(attention_weights)[-5:][::-1]

    print(f"   Top 5 most attended positions:")
    for idx in top_indices:
        if 0 < idx < len(sequence_chars) + 1:  # Skip special tokens
            aa_pos = idx - 1  # Adjust for CLS token
            if aa_pos < len(sequence_chars):
                amino_acid = sequence_chars[aa_pos]
                weight = attention_weights[idx]
                print(f"     Position {aa_pos+1}: {amino_acid} (weight: {weight:.4f})")

    print("\n✅ Example completed successfully!")
    print("\nTo use this model in your own code:")
    print("from model import create_model")
    print("model = create_model()")
    print(
        "result = model.predict('YOUR_SEQUENCE', {'Temperature (°C)': 37.0, 'pH': 7.4, 'Protein concentration (µM)': 50.0})"
    )


if __name__ == "__main__":
    main()
