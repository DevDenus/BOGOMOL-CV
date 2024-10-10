from random import getrandbits, randint

import numpy as np

from src.utils import bin_mat_mul


class BDenseLayer:
    """
    Binary dense layer
    nodes : int - an amount of nodes in layer
    output_nodes : int - an amount of nodes on output of the layer
    activation_function : callable - activation function of the layer (default is sign function)
    """
    def __init__(self, nodes : int, output_nodes : int = None, max_meta_parameter_size : int = None):
        self.nodes = nodes
        self.output_nodes = output_nodes
        self.weights = np.random.randint(0, 2, (self.output_nodes, self.nodes), dtype=np.int_)
        self.bias = np.random.randint(0, 2, (self.output_nodes), dtype=np.int_)
        self.max_meta_parameter_size = max_meta_parameter_size if max_meta_parameter_size else max(self.output_nodes//2, 1)
        self.trainable = True
        self.cached_x = None
        self.cached_weights = None
        self.cached_bias = None
        self.meta_parameters_indexes = None

    def generate_meta_parameters(self, meta_parameter_size : int = 1) -> list[np.array]:
        """
        Splits layer parameters into meta parameters, which consist
        of <meta_parameter_size> parameters
        """
        if self.meta_parameters_indexes:
            return self.meta_parameters_indexes
        meta_parameter_size = max(self.max_meta_parameter_size, meta_parameter_size)
        indices = np.arange(self.output_nodes)
        np.random.shuffle(indices)
        meta_parameters_indexes = []
        i = 0
        for i in range(self.output_nodes//meta_parameter_size):
            meta_parameters_indexes.append(indices[i*meta_parameter_size:(i+1)*meta_parameter_size])
        if i < len(self.weights):
            meta_parameters_indexes.append(indices[i*meta_parameter_size:])
        self.meta_parameters_indexes = meta_parameters_indexes
        return meta_parameters_indexes

    def __generate_annealing_mask(self) -> int:
        """
        Generates annealing mask
        """
        annealing_mask = np.random.randint(0, 2, self.nodes, dtype=np.int_)
        return annealing_mask

    def set_cached_weights(self):
        """
        Sets layer parameters to cached. Used for layer fit
        """
        assert self.trainable
        self.weights = self.cached_weights.copy()
        return self

    def set_cached_bias(self):
        """
        Sets layer bias to cached. Used for layer fit
        """
        assert self.trainable
        self.bias = self.cached_bias.copy()
        return self

    def anneal_weights(self, meta_parameter_index : int):
        """
        Implementation of annealing simulation.
        meta_parameters : int - if set weights is divided into <meta_parameters> subgroups
            and annealing is interfered over the subgroup.
        """
        assert meta_parameter_index < len(self.meta_parameters_indexes)
        if not self.trainable:
            return self
        self.cached_weights = self.weights.copy()
        for i in self.meta_parameters_indexes[meta_parameter_index]:
            annealing_mask = self.__generate_annealing_mask()
            self.weights[i] = np.logical_xor(self.weights[i], annealing_mask)
        return self

    def anneal_bias(self, meta_parameter_index : int):
        """
        Implementation of annealing simulation of bias.
        meta_parameters : int - if set weights is divided into <meta_parameters> subgroups
            and annealing is interfered over the subgroup.
        """
        assert meta_parameter_index < len(self.meta_parameters_indexes)
        if not self.trainable:
            return self
        self.cached_bias = self.bias.copy()
        for i in self.meta_parameters_indexes[meta_parameter_index]:
            self.bias[i] = np.random.randint(0, 2, dtype=np.int_)
        return self

    def get_cached_input(self):
        """
        Returns cached input of the layer
        """
        return self.cached_x

    def stop_training(self):
        """
        Stops the training process, empties cached values
        """
        self.trainable = False
        self.cached_weights = None
        self.cached_bias = None
        self.cached_x = None
        return self

    def predict(self, input_data : np.array) -> np.array:
        """
        Returns layer prediction for input_data
        input_data : np.array
        Using bitwise AND as an alternative of scalar vector multiplication
        """
        prediction = (bin_mat_mul(self.weights, input_data) + self.bias) % 2
        if self.trainable:
            self.cached_x = input_data
        return prediction

class BNormalizer:
    """
    Normalizes continuos data into a binary vector with <bit_representation> binary digits
    """
    def __init__(self, bit_representation : int):
        self.bit_representation = bit_representation
        self.data_delta = None
        self.data_mean = None
        self.trainable = True

    def fit(self, data : np.array):
        """
        Computes mean, max and min for further normalization tasks
        """
        assert self.trainable
        self.data_mean = np.mean(data, axis=0)
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        self.data_delta = np.linalg.norm(data_max - data_min, ord=np.inf) + 1e-6
        self.trainable = False
        return self

    # BUG: features representation order is wrong
    def __get_bit_representation(self, normalized_data : np.array) -> np.array:
        """
        Returns bit-vector representation of number in [0, 1) with size of self.bit_representation
        """
        assert self.bit_representation >= 1
        bit_representation = [np.int_(np.floor(normalized_data*2**i%2)) for i in range(1, self.bit_representation+1)]
        bit_representation = np.concat(bit_representation, axis=1, dtype=np.int_)
        return bit_representation

    def predict(self, data : np.array) -> np.array:
        """
        Normalizes data according to seen data and ensures elements are in [0, 1), using sigmoid
        function, then giving back a bit-vector of representation of the result
        """
        assert (self.data_delta is not None) and (self.data_mean is not None)
        normalized = (data - self.data_mean)/self.data_delta
        sigmoid_normalized = 1/(1 + np.exp(-normalized))
        return self.__get_bit_representation(sigmoid_normalized)
