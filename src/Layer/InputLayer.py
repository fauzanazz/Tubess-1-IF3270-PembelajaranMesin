class InputLayer:
    def __init__(self, input_size, layer_name="Input Layer"):
        self.input_size = input_size      
        self.num_neurons = input_size    
        self.layer_name = layer_name
        self.nodes = []  
        self.alpha = None  

    def forward(self, x, useRMSprop):
        return x

    def backward(self, lr, delta):
        return delta
