import math

def relu(x):
    return max(0, x)

def gradientclipping(x):
    return max(min(x, 1.0), -1.0)  

class RecurrentNeuralNetwork:
    def __init__(self, weights, features, bias):
        self.weights = weights  
        self.features = features 
        self.bias = bias  
        self.hidden_states = []  
        self.outputs = []  
        self.position = 0  

    def attention_mechanism(self, hidden_states):
        attention_data = []
        previous_delta = 0.001  
        for i in range(len(hidden_states) - 1):
            delta = hidden_states[i+1] - hidden_states[i]
            new_delta = delta ** 2 / (previous_delta + 1e-6)  
            previous_delta = new_delta
            attention_data.append(new_delta)
        normalized_data = sum(attention_data)
        return [x / normalized_data for x in attention_data]

    def forward(self, previous_feature=None, previous_output=None):
        if self.position >= len(self.features):
            return previous_output
        current_hidden_state = sum(
            self.weights[self.position][i] * self.features[self.position][i]
            for i in range(len(self.features[self.position]))
        ) + self.bias[self.position]
        self.hidden_states.append(current_hidden_state)
        if len(self.hidden_states) > 1:
            attention_weights = self.attention_mechanism(self.hidden_states)
            weighted_sum = sum(attention_weights[i] * self.hidden_states[i] for i in range(len(attention_weights)))
        else:
            weighted_sum = current_hidden_state
        self.position += 1
        previous_output = weighted_sum
        self.outputs.append(previous_output)
        return self.forward(previous_feature, previous_output)

    def backward(self, features_accumulated=None):
        if self.position <= 0:
            return features_accumulated
        bsum = sum(self.weights[self.position][i] * self.features[self.position][i] +
                   self.features[self.position][i] * self.bias[self.position]
                   for i in range(len(self.features[self.position])))
        self.position -= 1
        if features_accumulated is None:
            features_accumulated = []
        features_accumulated.append(bsum)
        return self.backward(features_accumulated)

    def backpropagation_structure(self, expected_output, learning_rate):
        for i in range(len(self.outputs)):
            delta = self.outputs[i] - expected_output
            gradient = delta / expected_output
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j] - learning_rate * gradient
            self.bias[i] = self.bias[i] - learning_rate * gradient

    def main(self):
        forward_features = self.forward(None, None)
        backward_features = self.backward([])
        self.backpropagation_structure(1.0, 0.01) 
