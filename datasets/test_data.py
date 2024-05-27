from synthetic_data import SyntheticData

import matplotlib.pyplot as plt

def test_synthetic_data():
    # Test the synthetic data generation
    data = SyntheticData(SyntheticData.DataType.LINEAR, 100, 0.1)
    print("Linear data shape:", data.data.head())
    assert data.data.shape == (100, 2)
    # plt.scatter(data.data.iloc[:, 0], data.data.iloc[:, 1], label="Linear")

    data = SyntheticData(SyntheticData.DataType.QUADRATIC, 100, 0.1)
    print("Quadratic data shape:", data.data.head())
    assert data.data.shape == (100, 2)
    # plt.scatter(data.data.iloc[:, 0], data.data.iloc[:, 1], label="Quadratic")

    data = SyntheticData(SyntheticData.DataType.CUBIC, 100, 0.1)
    print("Cubic data shape:", data.data.head())
    assert data.data.shape == (100, 2)
    # plt.scatter(data.data.iloc[:, 0], data.data.iloc[:, 1], label="Cubic")

    data = SyntheticData(SyntheticData.DataType.SINE, 500, 0.1)
    print("Sine data shape:", data.data.head())
    assert data.data.shape == (500, 2)
    #plt.scatter(data.data.iloc[:, 0], data.data.iloc[:, 1], label="Sine")

    data = SyntheticData(SyntheticData.DataType.ABSOLUTE, 100, 0.1)
    print("Absolute data shape:", data.data.head())
    assert data.data.shape == (100, 2)
    # plt.scatter(data.data.iloc[:, 0], data.data.iloc[:, 1], label="Absolute")

    data = SyntheticData(SyntheticData.DataType.EXPONENTIAL, 100, 0.1)
    print("Exponential data shape:", data.data.head())
    assert data.data.shape == (100, 2)
    # plt.scatter(data.data.iloc[:, 0], data.data.iloc[:, 1], label="Exponential")

    data = SyntheticData(SyntheticData.DataType.LOGARITHMIC, 1000, 0.3,10)
    print("Logarithmic data shape:", data.data.head())
    assert data.data.shape == (1000, 2)
    plt.scatter(data.data.iloc[:, 0], data.data.iloc[:, 1], label="Logarithmic")

    plt.legend()
    plt.show()

test_synthetic_data()