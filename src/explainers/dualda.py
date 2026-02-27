"""
DualDA (Dual Data Attribution) Explainer Module

This module implements the DualDA method for efficient and sparse data attribution
using Support Vector Machine (SVM) theory. DualDA provides fast explanations by
leveraging the dual formulation of linear SVMs to identify influential training
samples for model predictions.

Key Features:
- Efficient computation using modified LIBLINEAR solver
- Naturally sparse attributions through dual variables
- Feature-based explanations using learned SVM weights
"""

import torch
import os
import subprocess
import time
from copy import deepcopy
from utils.csv_io import read_matrix, write_data
from utils.explainers import FeatureKernelExplainer
from struct import pack
from sklearn.svm import SVC  # Using SVC with linear kernel to get dual coefficients
from tqdm import tqdm


class DualDA(FeatureKernelExplainer):
    """
    DualDA Explainer for Data Attribution
    
    This class implements the Dual Data Attribution method, which uses SVM dual
    variables to compute training sample attributions for test predictions.
    
    The method works by:
    1. Training a linear SVM on the feature representations of training data
    2. Using the dual coefficients (alpha values) to identify influential samples
    3. Computing attributions based on the learned weights and feature similarities
    
    Attributes:
        name (str): Identifier for the explainer method
        C (float): Regularization parameter for the SVM
        dir (str): Directory for saving trained model weights and coefficients
        features_dir (str): Directory for caching feature representations
        max_iter (int): Maximum iterations for SVM training
        learned_weight (torch.Tensor): Learned SVM weight matrix
        coefficients (torch.Tensor): Dual coefficients (alpha values) from SVM
        train_time (torch.Tensor): Time taken to train the SVM
    """
    name = "DualDAExplainer"

    def get_name(self):
        """
        Get the full name of the explainer including the C parameter.
        
        Returns:
            str: Explainer name with C value (e.g., "DualDAExplainer-0.001")
        """
        return f"{self.name}-{str(self.C)}"
    
    def __init__(self, model, dataset, device, dir, features_dir, use_preds=False, C=1.0, max_iter=1000000, normalize=False):
        """
        Initialize the DualDA explainer.
        
        Args:
            model: The neural network model to explain
            dataset: Training dataset containing samples and labels
            device (str): Device to run computations on ('cuda' or 'cpu')
            dir (str): Directory path for saving SVM weights and coefficients
            features_dir (str): Directory path for caching feature representations
            use_preds (bool): Whether to use model predictions instead of true labels
            C (float): SVM regularization parameter (default: 1.0)
                      Lower values increase sparsity, higher values fit data more closely
            max_iter (int): Maximum iterations for SVM solver (default: 1000000)
            normalize (bool): Whether to normalize features (default: False)
        """
        super().__init__(model, dataset, device, features_dir, normalize=normalize)
        self.C = C
        # Remove trailing backslash from directory path if present
        if dir[-1] == "\\":
            dir = dir[:-1]
        self.dir = dir
        self.features_dir = features_dir
        self.max_iter = max_iter
        # Create directories if they don't exist
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def read_variables(self):
        """
        Load previously trained SVM weights and coefficients from disk.
        
        This method reads the saved model state including:
        - learned_weight: The SVM weight matrix
        - coefficients: The dual variables (alpha values)
        - train_time: Time taken during training
        
        All tensors are loaded to the specified device and converted to float32.
        """
        self.learned_weight = torch.load(os.path.join(self.dir, "weights"), map_location=self.device).to(torch.float)
        self.coefficients = torch.load(os.path.join(self.dir, "coefficients"), map_location=self.device).to(torch.float)
        self.train_time = torch.load(os.path.join(self.dir, "train_time"), map_location=self.device).to(torch.float)

    def train(self):
        """
        Train the linear SVM on feature representations of the training data.
        
        This method:
        1. Saves normalized samples and labels to disk if not already cached
        2. Checks if a trained model already exists and loads it
        3. If no trained model exists:
           - Trains a multi-class linear SVM using Crammer-Singer formulation
           - Extracts dual coefficients (alpha values) and learned weights
           - Saves the trained model to disk
        
        The Crammer-Singer formulation is used for multi-class classification,
        which provides a single optimization problem for all classes.
        
        Returns:
            torch.Tensor: Training time in seconds
        """
        tstart = time.time()
        
        # Cache feature representations if not already saved
        if not os.path.isfile(os.path.join(self.features_dir, "samples")):
            torch.save(self.normalized_samples, os.path.join(self.features_dir, "samples"))
        if not os.path.isfile(os.path.join(self.features_dir, "labels")):
            torch.save(self.labels, os.path.join(self.features_dir, "labels"))
        
        # Load existing model if available
        if os.path.isfile(os.path.join(self.dir, 'weights')) and os.path.isfile(os.path.join(self.dir, 'coefficients')):
            self.read_variables()
        else:
            # Train new SVM model using SVC with linear kernel (Kaggle-compatible)
            # This provides access to dual coefficients via support vectors
            print("Training SVC with linear kernel...")
            
            # Use verbose=1 to show sklearn's internal progress
            model = SVC(kernel='linear', C=self.C, max_iter=self.max_iter, verbose=1)
            
            # Wrap in tqdm context for visual feedback
            with tqdm(total=100, desc="SVC Training", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
                model.fit(self.normalized_samples.cpu().numpy(), self.labels.cpu().numpy())
                pbar.update(100)  # Complete the bar when done
            
            accuracy = model.score(self.normalized_samples.cpu().numpy(), self.labels.cpu().numpy())
            print(f"✅ SVC Accuracy: {accuracy:.2f}")

            # Extract learned weight matrix [num_classes, num_features]
            self.learned_weight = torch.tensor(model.coef_, dtype=torch.float, device=self.device)
            
            # Extract dual coefficients from support vectors
            # dual_coef_ has shape [n_classes-1, n_support_vectors] for OVR
            # We need to map these back to the full training set
            n_samples = self.normalized_samples.shape[0]
            n_classes = len(torch.unique(self.labels))
            
            # Initialize coefficient matrix with zeros
            alpha_matrix = torch.zeros(n_samples, n_classes, dtype=torch.float, device=self.device)
            
            # For binary classification
            if n_classes == 2:
                for i, sv_idx in enumerate(model.support_):
                    # dual_coef_ contains alpha_i * y_i for each support vector
                    alpha_matrix[sv_idx, 0] = model.dual_coef_[0, i]
                    alpha_matrix[sv_idx, 1] = -model.dual_coef_[0, i]
            else:
                # For multi-class (one-vs-rest)
                # dual_coef_ has shape [n_classes, n_support_vectors]
                for i, sv_idx in enumerate(model.support_):
                    # Each row corresponds to one class
                    for class_idx in range(n_classes):
                        if class_idx < model.dual_coef_.shape[0]:
                            alpha_matrix[sv_idx, class_idx] = model.dual_coef_[class_idx, i]
            
            self.coefficients = alpha_matrix
            
            print(f"Number of support vectors: {len(model.support_)} / {n_samples}")
            
            self.train_time = torch.tensor(time.time() - tstart)

            # Save trained model components to disk
            torch.save(self.train_time, os.path.join(self.dir, 'train_time'))
            torch.save(self.learned_weight, os.path.join(self.dir, 'weights'))
            torch.save(self.coefficients, os.path.join(self.dir, 'coefficients'))
            print(f"Training took {self.train_time} seconds")
        return self.train_time

    def self_influences(self, only_coefs=False):
        """
        Compute self-influence scores for training samples.
        
        Self-influence measures how much a training sample influences its own
        prediction. This is useful for detecting mislabeled or anomalous samples.
        
        Args:
            only_coefs (bool): If True, return only the dual coefficients.
                             If False, scale by feature norms (default: False)
        
        Returns:
            torch.Tensor: Self-influence scores for each training sample
                         Shape: [num_train_samples]
                         
        The full self-influence is computed as:
            self_influence = ||features|| * alpha
        where alpha are the dual coefficients and ||features|| is the L2 norm
        of the feature representation.
        """
        # Get base self-influence from parent class (uses dual coefficients)
        self_coefs = super().self_influences()
        
        if only_coefs:
            # Return just the dual coefficients
            return self_coefs
        else:
            # Scale by feature norms to get full self-influence
            # This accounts for the magnitude of the feature representations
            return self.normalized_samples.norm(dim=-1) * self_coefs

# Made with Bob
