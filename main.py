import math
from math import log10

def relu(x):
    return max(x, 0)
def gradientclipping(x):
    return x ** -1
class RecurrentNerualNetwork:
    weights: [[int]]
    features: [[int]]
    bias: [[int]]
    asum: int
    outputs: []
    attention_weights: []
    position: int
    def attention_mechanism(self, hidden_states, outputs):
        attention_data = []
        for i in range(len(hidden_states) - 1):
            delta = hidden_states[i+1] - hidden_states[i]
            new_delta = delta ** 2/ (previous_delta + 1e-6)
            previous_delta = new_delta
            attention_data.append(new_delta)
        normalizeddata = sum(attention_data)
        return [x / normalizeddata for x in attention_data]

    def forward(self, previous_feature, previous_output):
        if self.position >= len(self.features):
            return previous_output
        current_hidden_state = sum(self.weights[self.position] * self.features[self.position]) + self.features[
            self.position] * self.bias[self.position]
        attention_weights = self.attention_mechanism(self.hidden_states, current_hidden_state)
        weighted_sum = sum(attention_weights[i] * self.hidden_states[i] for i in range(len(attention_weights)))
        self.position += 1
        previous_feature = self.features[self.position - 1]
        previous_output = weighted_sum
        self.outputs.append(previous_output)
        self.hidden_states.append(current_hidden_state)
        return self.forward(self, previous_feature, previous_output)
    def backward(self, feature):
        if self.position <= 0:
            return feature
        bsum = sum(self.weights[self.position] * self.features[self.position] + self.features[self.position] * self.bias[self.position])
        self.position-=1
        feature.append(bsum)
        self.backward(self, feature)
    def backpropgationstructure(self, output, learning_rate):
        for i in range(len(output)):
            delta = self.output[i] - output
            gradient = delta / output
            self.weights[i] = self.weights[i] - learning_rate * gradient
            self.bias[i] = self.bias[i] - learning_rate * gradient
    def main(self):
        forward_features = self.forward(self, None, None)
        backward_features = self.backward(self, [])
