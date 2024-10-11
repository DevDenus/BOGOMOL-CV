import numpy as np

from src.utils import bin_matmul

# TODO: Replace polynomial coefficients matrix of (dimension_size+1)^2
# elements with some smarter data structure with dimension_size*(dimension_size+1)/2 + 1
# elements and remove appropriate coefficient mask
class SecondDegreeZhegalkinPolynomial:
    """
    Implementation of Zhegalkin binary polynomial
    of second degree
    """
    def __init__(self, dimension_size : int) -> None:
        self.dimension_size = dimension_size
        self.polynomial_coefficients = np.random.randint(
            0, 2, (self.dimension_size+1, self.dimension_size+1), dtype=np.bool_
        )
        self.appropriate_coefficients_mask = np.zeros(
            ((self.dimension_size+1, self.dimension_size+1)), dtype=np.bool_
        )
        for i in range(self.dimension_size):
            self.appropriate_coefficients_mask[i, i:self.dimension_size] = True
        self.appropriate_coefficients_mask[self.dimension_size,self.dimension_size] = True
        self.polynomial_coefficients = np.logical_and(
            self.polynomial_coefficients, self.appropriate_coefficients_mask
        )
        self.trainable = True
        self.cached_coefficients = None

    def __call__(self, x : np.array) -> np.array:
        """
        Computes the value of Zhegalkin polynomial in given point x
        (or points in case of x is a binary matrix)
        x : np.array - binary vector or matrix
        """
        assert x.shape[1] == self.dimension_size, "Dimension of input must be the same as polynomial dimension!"
        # Used to store constant term of polynomial in polynomials coefficient matrix
        x = np.concat([x, np.ones((x.shape[0], 1))], axis=1)
        x = np.array(x, dtype=np.bool_)     # dtype=np.bool_ raise an exception in concat
        result = np.concat(
            [bin_matmul(bin_matmul(x[i].reshape(1, -1), self.polynomial_coefficients), x[i].reshape(1, -1).T) for i in range(x.shape[0])],
            axis=0
        )
        result = np.array(result, dtype=np.bool_)
        return result

    def __str__(self) -> str:
        """
        String representation of Zhegalkin polynomial
        """
        result = ""
        for i in range(self.dimension_size):
            for j in range(i + 1, self.dimension_size):
                result += f"{int(self.polynomial_coefficients[i,j])}&x_{i+1}&x_{j+1} XOR "
        for i in range(self.dimension_size):
            result += f"{int(self.polynomial_coefficients[i, i])}&x_{i+1} XOR "
        result += f"{int(self.polynomial_coefficients[self.dimension_size, self.dimension_size])}"
        return result

    def anneal_coefficients(self):
        """
        Implementation of annealing simulation on polynomial coefficients
        """
        if self.trainable:
            self.cached_coefficients = self.polynomial_coefficients.copy()
        self.polynomial_coefficients = np.random.randint(
            0, 2, (self.dimension_size+1, self.dimension_size+1), dtype=np.bool_
        )
        self.polynomial_coefficients = np.logical_and(
            self.polynomial_coefficients, self.appropriate_coefficients_mask
        )
        return self

    def set_cached_coefficients(self):
        """
        Sets coefficients to previously cached ones
        """
        assert not self.cached_coefficients is None, "Coefficients were not cached!"
        self.polynomial_coefficients = self.cached_coefficients.copy()
        return self

    def stop_training(self):
        """
        Clears cache
        """
        self.trainable = False
        self.cached_coefficients = None
        return self
