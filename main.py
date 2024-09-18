import math
from math import log10

def relu(x):
    return max(x, 0)

def gradientclipping(x):
    return x ** -1

class RecurrentNerualNetwork:
    def __init__(self, weights, features, bias, learning_rate):
        self.weights = weights
        self.features = features
        self.bias = bias
        self.asum = 0
        self.outputs = []
        self.attention_weights = []
        self.position = 0
        self.hidden_states = []
        self.learning_rate = learning_rate

    def attention_mechanism(self, hidden_states, current_hidden_state):
        attention_data = []
        for prev_hidden_state in hidden_states:
            attention_value = 0
            for i in range(len(prev_hidden_state)):
                attention_value += prev_hidden_state[i] * current_hidden_state[i]
            attention_data.append(attention_value)
        normalized_data = sum(attention_data)
        return [x / normalized_data for x in attention_data]

    def forward(self, previous_feature, previous_output):
        if self.position >= len(self.features):
            return previous_output

        current_hidden_state = []
        for i in range(len(self.weights[self.position])):
            weighted_sum = 0
            for j in range(len(self.weights[self.position][i])):
                weighted_sum += self.weights[self.position][i][j] * self.features[self.position][j]
            current_hidden_state.append(weighted_sum + self.features[self.position][i] * self.bias[self.position][i])

        attention_weights = self.attention_mechanism(self.hidden_states, current_hidden_state)
        weighted_sum = 0
        for i in range(len(attention_weights)):
            weighted_sum += attention_weights[i] * self.hidden_states[i]

        self.position += 1
        previous_feature = self.features[self.position - 1]
        previous_output = weighted_sum
        self.outputs.append(previous_output)
        self.hidden_states.append(current_hidden_state)
        return self.forward(self, previous_feature, previous_output)

    def backward(self, output):
        gradients = []
        for i in range(len(self.outputs) - 1, -1, -1):
            delta = self.outputs[i] - output
            gradient = delta / output
            gradients.append(gradient)
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j] - self.learning_rate * gradient
                self.bias[i][j] = self.bias[i][j] - self.learning_rate * gradient

        attention_gradients = []
        for i in range(len(self.attention_weights) - 1, -1, -1):
            delta = self.attention_weights[i] - output
            gradient = delta / output
            attention_gradients.append(gradient)
            self.attention_weights[i] = self.attention_weights[i] - self.learning_rate * gradient

        gradients.reverse()
        attention_gradients.reverse()
        return gradients, attention_gradients

    def train(self, input_data, target_output, num_epochs):
        for epoch in range(num_epochs):
            self.forward(input_data, None)
            gradients, attention_gradients = self.backward(target_output)
        

    def predict(self, input_data):
        return self.forward(input_data, None)

if __name__ == "__main__":
    pass
