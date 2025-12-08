"""
Classification Engine (using precomputed .npy embeddings)
=========================================================
"""

import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class ClassificationEngine:
    def __init__(
        self,
        embedding_path: str = "ComputerVisionFiles/fashion_mnist_resnet50_embeddings.npy",
        label_path: str = "ComputerVisionFiles/fashion_mnist_labels.npy",
    ):
        """
        Loads all embeddings and labels into memory, and trains the classifier once.
        """

        # -----------------------------------------------------------
        # Load training data from .npy files provided by the user
        # -----------------------------------------------------------
        try:
            self.X = np.load(embedding_path)  # shape: (N, D)
            self.y_raw = np.load(label_path)  # shape: (N,)
        except FileNotFoundError:
            raise RuntimeError(
                f"Could not find embeddings or label files at: "
                f"{embedding_path}, {label_path}"
            )

        if self.X.shape[0] != self.y_raw.shape[0]:
            raise RuntimeError(
                "Embedding count does not match label count. "
                f"{self.X.shape[0]} embeddings vs {self.y_raw.shape[0]} labels."
            )

        # -----------------------------------------------------------
        # Encode labels
        # -----------------------------------------------------------
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y_raw)

        # -----------------------------------------------------------
        # Train classifier (Logistic Regression)
        # -----------------------------------------------------------
        self.model = LogisticRegression(max_iter=3000)
        self.model.fit(self.X, self.y)

        print("ClassificationEngine initialized.")
        print(f"Loaded {self.X.shape[0]} embeddings.")
        print(f"Classes: {self.label_encoder.classes_.tolist()}")


    # ---------------------------------------------------------------------
    # PREDICT A LABEL FOR A NEW EMBEDDING
    # ---------------------------------------------------------------------
    def predict(self, embedding: np.ndarray) -> Dict:
        """
        Predicts the class label for a new embedding.

        Parameters
        ----------
        embedding : np.ndarray
            A vector of shape (D,) or (1, D)

        Returns
        -------
        Dict containing:
            label       → predicted class label
            confidence  → model confidence in that prediction
        """

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)  # shape → (1, D)

        y_pred = self.model.predict(embedding)
        y_proba = self.model.predict_proba(embedding)[0]

        label = self.label_encoder.inverse_transform(y_pred)[0]
        confidence = float(np.max(y_proba))

        return {
            "label": label,
            "confidence": confidence,
        }
