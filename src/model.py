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
        batch_features, batch_targets = [], []
        batch_x, batch_y = [], []
        for i in indices:
            batch_x.append(input_data[i])
            batch_y.append(labels[i])
            if len(batch_x) == batch_size:
                batch_features.append(np.array(batch_x))
                batch_targets.append(np.array(batch_y))
                batch_x, batch_y = [], []
        if batch_x and batch_y:
            batch_features.append(np.array(batch_x))
            batch_targets.append(np.array(batch_y))
        return batch_features, batch_targets

    # FIXME Instability of learning discovered, possibly
    # it has to be related to temperature
    def __anneal_backward_pass(self, input_data : np.array, labels : np.array, meta_parameter_size : int, temperature : float) -> float:
        """
        Realization of annealing stochastic learning method.
        input_data : np.array - training data
        labels : np.array - training labels
        loss_fn : callable - function used to count loss
        meta_parameter_size : int - size of meta_parameter in layer
        """
        prediction = self.predict(input_data)
        loss = self.loss_fn(prediction, labels)
        for layer in self.layers[::-1]:
            if not layer.trainable:
                continue
            meta_parameters = layer.generate_meta_parameters(meta_parameter_size)
            for meta_parameter_index in range(len(meta_parameters)):
                layer.anneal(meta_parameter_index)
                prediction = self.predict(input_data)
                new_loss = self.loss_fn(prediction, labels)
                if np.mean(loss, axis=0) <= np.mean(new_loss, axis=0):
                    transition_probability = float(np.exp(-np.mean(new_loss-loss)/temperature))
                    if float(np.random.uniform(0, 1)) < transition_probability:
                        loss = new_loss
                    else:
                        layer.set_cached_weights()
                else:
                    loss = new_loss
        print(np.mean(loss))

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
            "loss" : [],
            "accuracy" : [],
            "precision" : [],
            "recall" : []
        }
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}:")
            batch_features, batch_labels = self.__make_batches(input_data, labels, batch_size)
            for batch_feature, batch_label in tqdm(zip(batch_features, batch_labels)):
                history['loss'].append(self.__anneal_backward_pass(batch_feature, batch_label, meta_parameter_size, 1/epoch))
        return history

    def stop_training(self):
        """
        Stops training, empties cached values
        """
        for layer in self.layers:
            layer.stop_training()
        return self

    def predict(self, input_data : np.array) -> np.array:
        """
        Interferes a model
        input_data : np.array - data
        """
        result = input_data
        for layer in self.layers:
            result = layer.predict(result)
        return result
