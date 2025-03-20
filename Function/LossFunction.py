import torch
import numpy as np

class LossFunction:
    @staticmethod
    def mean_squared_error(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    @torch.compile
    def mean_squared_error_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def binary_cross_entropy(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    @torch.compile
    def binary_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return ((y_pred - y_true) / (y_pred * (1 - y_pred))) / y_true.shape[0]

    @staticmethod
    def categorical_cross_entropy(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Handle different dimensions
        if y_pred.ndim == 1:
            # For 1D arrays (binary case)
            y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # For 2D arrays (multi-class case)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    @torch.compile
    def categorical_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        
        # Handle different dimensions
        if y_pred.ndim == 1:
            # For binary case
            return ((y_pred - y_true) / (y_pred * (1 - y_pred))) / y_true.shape[0]
        else:
            # For multi-class case
            return -(y_true / y_pred) / y_pred.shape[0]

if __name__ == "__main__":
    # Binary test data
    y_true_binary = np.array([1, 0, 0, 1, 1], dtype=np.float32)
    y_pred_binary = np.array([0.9, 0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    
    # PyTorch tensors data
    y_true_binary_torch = torch.tensor(y_true_binary, dtype=torch.float32)
    y_pred_binary_torch = torch.tensor(y_pred_binary, dtype=torch.float32)
    y_pred_binary_torch.requires_grad = True
    
    # Multi-class test data
    y_true_multi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    y_pred_multi = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.3, 0.6, 0.1]
    ], dtype=np.float32)
    
    # PyTorch multi-class tensors
    y_true_multi_torch = torch.tensor(y_true_multi, dtype=torch.float32)
    y_pred_multi_torch = torch.tensor(y_pred_multi, dtype=torch.float32)
    y_pred_multi_torch.requires_grad = True
    
    # Test MSE derivatives
    print("MSE Derivatives:")
    mse_deriv_np = LossFunction.mean_squared_error_derivative(y_pred_binary, y_true_binary)
    
    # PyTorch native implementation using autograd
    mse_loss = torch.nn.functional.mse_loss(y_pred_binary_torch, y_true_binary_torch)
    mse_loss.backward()
    mse_deriv_torch_native = y_pred_binary_torch.grad.clone()
    y_pred_binary_torch.grad.zero_()
    
    print(f"  NumPy result:              {mse_deriv_np}")
    print(f"  PyTorch autograd result:   {mse_deriv_torch_native.numpy()}")
    print(f"  Max difference:            {np.max(np.abs(mse_deriv_np - mse_deriv_torch_native.numpy()))}")
    
    # Test BCE derivatives
    print("\nBCE Derivatives:")
    bce_deriv_np = LossFunction.binary_cross_entropy_derivative(y_pred_binary, y_true_binary)
    
    # PyTorch native implementation using autograd
    bce_loss = torch.nn.functional.binary_cross_entropy(y_pred_binary_torch, y_true_binary_torch)
    bce_loss.backward()
    bce_deriv_torch_native = y_pred_binary_torch.grad.clone()
    y_pred_binary_torch.grad.zero_()
    
    print(f"  NumPy result:              {bce_deriv_np}")
    print(f"  PyTorch autograd result:   {bce_deriv_torch_native.numpy()}")
    print(f"  Max difference:            {np.max(np.abs(bce_deriv_np - bce_deriv_torch_native.numpy()))}")
    
    # Test CCE derivatives (multi-class)
    print("\nCCE Derivatives (Multi-class):")
    cce_deriv_np = LossFunction.categorical_cross_entropy_derivative(y_pred_multi, y_true_multi)
    
    # PyTorch native implementation using autograd
    y_true_indices = torch.argmax(y_true_multi_torch, dim=1)
    cce_loss = torch.nn.functional.cross_entropy(y_pred_multi_torch, y_true_indices)
    cce_loss.backward()
    cce_deriv_torch_native = y_pred_multi_torch.grad.clone()
    y_pred_multi_torch.grad.zero_()
    
    print(f"  NumPy shape:               {cce_deriv_np.shape}")
    print(f"  PyTorch autograd shape:    {cce_deriv_torch_native.shape}")
    