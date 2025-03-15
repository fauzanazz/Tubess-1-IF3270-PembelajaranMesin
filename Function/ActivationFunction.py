import torch
import torch.nn.functional as F
class ActivationFunction:
    """
    ActivationFunction Class.

    This class is used to define the activation function.
    """

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    @torch.compile
    def relu(x):
        return torch.relu(x)

    @staticmethod
    @torch.compile
    def sigmoid(x):
        return torch.sigmoid(x)

    @staticmethod
    @torch.compile
    def tanh(x):
        return torch.tanh(x)

    @staticmethod
    @torch.compile
    def softmax(x):
        x = x - torch.max(x)
        return torch.softmax(x, dim=0)

    @staticmethod
    def derivative_linear(x):
        return torch.ones_like(x)

    @staticmethod
    @torch.compile
    def derivative_relu(x):
        return (x > 0).float()

    @staticmethod
    @torch.compile
    def derivative_sigmoid(x):
        sig = torch.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    @torch.compile
    def derivative_tanh(x):
        return 1 - torch.tanh(x) ** 2

    @staticmethod
    @torch.compile
    def derivative_softmax(x):
        def softmax(z):
            return torch.exp(z) / torch.sum(torch.exp(z), dim=-1, keepdim=True)
        sm = softmax(x).unsqueeze(-1)  # Shape: [N, 1]
        jacobian = torch.diag_embed(sm.squeeze(-1)) - torch.matmul(sm, sm.transpose(-1, -2))
        return jacobian

if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    x_softmax = torch.tensor([[4., 2.]], requires_grad=True)

    print("Linear")
    print(ActivationFunction.linear(x))
    print(x)

    print("ReLU")
    print(ActivationFunction.relu(x))
    print(torch.nn.functional.relu(x))

    print("Sigmoid")
    print(ActivationFunction.sigmoid(x))
    print(torch.nn.functional.sigmoid(x))

    print("Tanh")
    print(ActivationFunction.tanh(x))
    print(torch.nn.functional.tanh(x))

    print("Softmax")
    print(ActivationFunction.softmax(x))
    print(torch.nn.functional.softmax(x, dim=0))

    print("Derivative Linear")
    print(ActivationFunction.derivative_linear(x))
    print(torch.ones_like(x))

    print("Derivative ReLU")
    print(ActivationFunction.derivative_relu(x))
    t = x.clone().detach().requires_grad_(True)
    t.relu().sum().backward()
    print(t.grad)

    print("Derivative Sigmoid")
    print(ActivationFunction.derivative_sigmoid(x))
    t = x.clone().detach().requires_grad_(True)
    t.sigmoid().sum().backward()
    print(t.grad)

    print("Derivative Tanh")
    print(ActivationFunction.derivative_tanh(x))
    t = x.clone().detach().requires_grad_(True)
    t.tanh().sum().backward()
    print(t.grad)

    print("Derivative Softmax")
    print(ActivationFunction.derivative_softmax(x_softmax))
    torch_sm = F.softmax(x_softmax, dim=1)
    # to extract the first row in the jacobian matrix, use [[1., 0]]
    # retain_graph=True because we re-use backward() for the second row
    torch_sm.backward(torch.tensor([[1.,0.]]), retain_graph=True)
    r1 = x_softmax.grad
    x_softmax.grad = torch.zeros_like(x_softmax)
    # to extract the second row in the jacobian matrix, use [[0., 1.]]
    torch_sm.backward(torch.tensor([[0.,1.]]))
    r2 = x_softmax.grad
    torch_sm_p = torch.cat((r1,r2))
    print(torch_sm_p)

