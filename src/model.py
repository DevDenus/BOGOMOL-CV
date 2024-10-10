import numpy as np

from tqdm import tqdm

class BModel:
    """
    Sequential binary model
    """
    def __init__(self, layers : list[object], loss_fn : callable):
        self.layers = layers
        self.loss_fn = loss_fn

    def __make_batches(self, input_data : np.array, labels : np.array, batch_size : int) -> np.array:
        """
        Splits dataset into batched of size <batch_size>
        input_data : np.array - training data
        labels : np.array - training labels
        batch_size : int - size of batch
        """
        assert input_data.shape[0] == labels.shape[0]
        assert input_data.shape[0] >= batch_size
        n_samples = input_data.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        batch_features = []
        batch_targets = []

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            batch_features.append(input_data[batch_indices])
            batch_targets.append(labels[batch_indices])

        return batch_features, batch_targets

    # FIXME Instability of learning discovered, possibly
    # it has to be related to temperature
    def __anneal_backward_pass(self, input_data : np.array, labels : np.array, meta_parameter_size : int, epoch : int) -> float:
        """
        Realization of annealing stochastic learning method.
        input_data : np.array - training data
        labels : np.array - training labels
        loss_fn : callable - function used to count loss
        meta_parameter_size : int - size of meta_parameter in layer
        """
        prediction = self.predict(input_data)
        loss = self.loss_fn(prediction, labels)
        for layer_index in range(len(self.layers)-1, -1, -1):
            if not self.layers[layer_index].trainable:
                continue
            meta_parameters = self.layers[layer_index].generate_meta_parameters(meta_parameter_size)
            for meta_parameter_index in range(len(meta_parameters)):
                self.layers[layer_index].anneal_weights(meta_parameter_index)
                prediction = self.__predict_from_cached(layer_index)
                new_loss = self.loss_fn(prediction, labels)
                #print(1, loss - new_loss)
                if loss < new_loss:
                    # temperature = 1 / (1 + epoch*np.log2(1 + epoch))
                    temperature = np.pow(0.9, epoch)
                    transition_probability = np.exp(-(new_loss-loss)/temperature)
                    prob = np.random.uniform(0, 1)
                    # print(transition_probability, prob, prob < transition_probability)
                    if prob < transition_probability:
                        loss = new_loss
                    else:
                        self.layers[layer_index].set_cached_weights()
                else:
                    loss = new_loss

                self.layers[layer_index].anneal_bias(meta_parameter_index)
                prediction = self.__predict_from_cached(layer_index)
                new_loss = self.loss_fn(prediction, labels)
                #print(2, loss - new_loss)
                if loss < new_loss:
                    #temperature = 1 / (1 + epoch * np.log2(1 + epoch))
                    temperature = np.pow(0.9, epoch)
                    transition_probability = np.exp(-(new_loss-loss)/temperature)
                    prob = np.random.uniform(0, 1)
                    if prob < transition_probability:
                        loss = new_loss
                    else:
                        self.layers[layer_index].set_cached_bias()
                else:
                    loss = new_loss
        return loss

    def fit(self, input_data : np.array, labels : np.array, batch_size : int = 1,
            meta_parameter_size : int = 1, epochs : int = 1) -> dict:
        """
        Model fit method.
        input_data : np.array - training data
        labels : np.array - training labels
        loss_fn : callable - function used to count loss of the model
        batch_size : int - size of batch
        epochs : int - number of training epochs
        meta_parameter_size : int - size of meta_parameter in layer
        """
        history = {
            "loss" : [float('inf')],
            "accuracy" : [0],
            "precision" : [0],
            "recall" : [0]
        }
        for epoch in range(1, epochs+1):
            batch_features, batch_labels = self.__make_batches(input_data, labels, batch_size)
            for batch_feature, batch_label in tqdm(zip(batch_features, batch_labels), desc=f'epoch : {epoch} loss : {history["loss"][-1]}'):
                history['loss'].append(self.__anneal_backward_pass(batch_feature, batch_label, meta_parameter_size, epoch))
        self.stop_training()
        return history

    def stop_training(self):
        """
        Stops training, empties cached values
        """
        for layer in self.layers:
            layer.stop_training()
        return self

    def __predict_from_cached(self, layer_index : int) -> np.array:
        result = self.layers[layer_index].get_cached_input()
        for i in range(layer_index, len(self.layers)):
            result = self.layers[i].predict(result)
        return result

    def predict(self, input_data : np.array) -> np.array:
        """
        Interferes a model
        input_data : np.array - data
        """
        result = input_data
        for layer in self.layers:
            result = layer.predict(result)
        return result
