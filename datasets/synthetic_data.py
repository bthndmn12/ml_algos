import numpy as np
import pandas as pd
from enum import Enum


class SyntheticData():
    # enums for data types
    class DataType(Enum):
        LINEAR = 1
        QUADRATIC = 2
        CUBIC = 3
        SINE = 4
        ABSOLUTE = 5
        EXPONENTIAL = 6
        LOGARITHMIC = 7
    
    def __init__(self, data_type, num_points, noise=0, space = 1):
        if not isinstance(data_type, SyntheticData.DataType):
            raise ValueError("Invalid data type")
        self.data_type = data_type
        self.num_points = num_points
        self.noise = noise
        self.space = space
        self.data = None
        self.generate_data()

    
    def generate_data(self):
        if self.data_type == self.DataType.LINEAR:
            self.data = self.generate_linear_data()
        elif self.data_type == self.DataType.QUADRATIC:
            self.data = self.generate_quadratic_data()
        elif self.data_type == self.DataType.CUBIC:
            self.data = self.generate_cubic_data()
        elif self.data_type == self.DataType.SINE:
            self.data = self.generate_sine_data()
        elif self.data_type == self.DataType.ABSOLUTE:
            self.data = self.generate_absolute_data()
        elif self.data_type == self.DataType.EXPONENTIAL:
            self.data = self.generate_exponential_data()
        elif self.data_type == self.DataType.LOGARITHMIC:
            self.data = self.generate_logarithmic_data()

    def generate_linear_data(self):
        x = np.linspace(0, self.space, self.num_points)
        y = 2 * x + 1 + np.random.normal(0, self.noise, self.num_points)
        return pd.DataFrame({'x': x, 'y': y})
    
    def generate_quadratic_data(self):
        x = np.linspace(0, self.space, self.num_points)
        y = 3 * x ** 2 + 2 * x + 1 + np.random.normal(0, self.noise, self.num_points)
        return pd.DataFrame({'x': x, 'y': y})
    
    def generate_cubic_data(self):
        x = np.linspace(0, self.space, self.num_points)
        y = 4 * x ** 3 + 3 * x ** 2 + 2 * x + 1 + np.random.normal(0, self.noise, self.num_points)
        return pd.DataFrame({'x': x, 'y': y})
    
    def generate_sine_data(self):
        x = np.linspace(0, self.space, self.num_points)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, self.noise, self.num_points)
        return pd.DataFrame({'x': x, 'y': y})
    
    def generate_absolute_data(self):
        x = np.linspace(-1*self.space, self.space, self.num_points)
        y = np.abs(x) + np.random.normal(0, self.noise, self.num_points)
        return pd.DataFrame({'x': x, 'y': y})
    
    def generate_exponential_data(self):
        x = np.linspace(0, self.space, self.num_points)
        y = np.exp(x) + np.random.normal(0, self.noise, self.num_points)
        return pd.DataFrame({'x': x, 'y': y})
    
    def generate_logarithmic_data(self):
        x = np.linspace(0.1, self.space, self.num_points)
        y = -np.log(x) + np.random.normal(0, self.noise, self.num_points)
        return pd.DataFrame({'x': x, 'y': y})