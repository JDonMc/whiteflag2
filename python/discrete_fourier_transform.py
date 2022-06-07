import math
import matplotlib.pyplot as plt
import numpy as np

class CircularNeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (math.sqrt(1+np.exp(-x)**2))

    def _sigmoid_deriv(self, x):
        return math.sqrt(100+(self._sigmoid(x) * (1 - self._sigmoid(x)))**2)-math.sqrt(100)
        #return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _sigmoid2(self, x):
        return 1 / (1 + np.exp(-x+2))

    def _sigmoid2_deriv(self, x):
        return math.sqrt(100+(self._sigmoid2(x) * (1 - self._sigmoid2(x)))**2)-math.sqrt(100)
        #return self._sigmoid(x) * (1 - self._sigmoid(x))


    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors


def CircularDFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.sqrt(1+ (-4j*np.pi*k *np.exp(-4j * np.pi * k * n / N))**2)
    X = np.dot(e, x)
    return X  


def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-4j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X    

# sampling rate
sr = 100
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)

freq = 4
x += np.sin(2*np.pi*freq*t)

freq = 7   
x += 0.5* np.sin(2*np.pi*freq*t)

plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')

plt.show()


X = DFT(x)

# calculate the frequency
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (8, 6))
plt.stem(freq, abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.show()

Y = CircularDFT(x)

# calculate the frequency
N = len(Y)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (8, 6))
plt.stem(freq, abs(Y), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.show()


class Wavelet:
    def __init__(self, wavelet_array_length):
        self.wavelet_array_length = wavelet_array_length
        self.max_level = round(math.log(wavelet_array_length) / math.log(2.))
        self.transform_wave_length = 2
        self.mother_wave_length = 18
        self.scaling_de_com = [None]*self.mother_wave_length
        self.scaling_de_com[0] = 0.0014009155259146807
        self.scaling_de_com[1] = 0.0006197808889855868
        self.scaling_de_com[2] = -0.013271967781817119
        self.scaling_de_com[3] = -0.01152821020767923
        self.scaling_de_com[4] = 0.03022487885827568
        self.scaling_de_com[5] = 0.0005834627461258068
        self.scaling_de_com[6] = -0.05456895843083407
        self.scaling_de_com[7] = 0.238760914607303
        self.scaling_de_com[8] = 0.717897082764412
        self.scaling_de_com[9] = 0.6173384491409358
        self.scaling_de_com[10] = 0.035272488035271894
        self.scaling_de_com[11] = -0.19155083129728512
        self.scaling_de_com[12] = -0.018233770779395985
        self.scaling_de_com[13] = 0.06207778930288603
        self.scaling_de_com[14] = 0.008859267493400484
        self.scaling_de_com[15] = -0.010264064027633142
        self.scaling_de_com[16] = -0.0004731544986800831
        self.scaling_de_com[17] = 0.0010694900329086053
        self.wavelet_de_com = [None]*18
        self.wavelet_re_con = [None]*18
        self.scaling_re_con = [None]*18
        self.build_orthonormal_space()

    def forward(self, time_points): # how do I include the level?
        arr_hilbert = [None]*len(time_points)
        h = self.wavelet_array_length >> 1
        for i in range(0, h, 1):
            arr_hilbert[i] = 0.  # why is this setting to zero again?
            arr_hilbert[i+h] = 0.
            for j in range(0, self.mother_wave_length, 1):
                k = i << 1
                while k >= self.wavelet_array_length:
                    k -= self.wavelet_array_length
                arr_hilbert[i] += time_points[k] * self.scaling_de_com[j]
                arr_hilbert[i+h] += time_points[k] * self.wavelet_de_com[j]
        return arr_hilbert

    def reverse(self, hilbert_points):
        arr_time = [None]*len(hilbert_points)
        for i in range(0, self.wavelet_array_length, 1):
            arr_time[i] = 0.
        h = self.wavelet_array_length >> 1
        for i in range(0, h, 1):
            for j in range(0, self.mother_wave_length, 1):
                k = (i << 1) + j
                while k >= self.wavelet_array_length:
                    k -= self.wavelet_array_length
                arr_time[k] += (hilbert_points[i] * self.scaling_re_con[j]) + \
                               (hilbert_points[i+h] * self.wavelet_re_con[j])
        return arr_time

    def get_scaling_length(self):
        return self.mother_wave_length

    def set_scaling_length(self, scale_length):
        self.mother_wave_length = scale_length
        self.scaling_de_com = [None]*self.mother_wave_length

    def save_scaling(self):
        return self.scaling_de_com

    def set_scaling(self, scaling):
        for s in range(0, self.mother_wave_length, 1):
            self.scaling_de_com[s] = scaling[s]
        self.build_orthonormal_space()

    def build_orthonormal_space(self):
        self.wavelet_de_com = [None]*self.mother_wave_length
        for i in range(0, self.mother_wave_length, 1):
            if i % 2 == 0:
                self.wavelet_de_com[i] = self.scaling_de_com[self.mother_wave_length-1-i]
            else:
                self.wavelet_de_com[i] = -self.scaling_de_com[self.mother_wave_length - 1 - i]
        self.scaling_re_con = [None]*self.mother_wave_length
        self.wavelet_re_con = [None]*self.mother_wave_length
        for i in range(0, self.mother_wave_length, 1):
            self.scaling_re_con[i] = self.scaling_de_com[i]
            self.wavelet_re_con[i] = self.wavelet_de_com[i]

wavelet = Wavelet(len(x))
Z = wavelet.forward(x)
plt.figure(figsize = (8, 6))
plt.stem(freq, np.abs(Z), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.show()


