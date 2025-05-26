import numpy as np

def load_predictor_dataset():
    """
    Returns dummy data for predictor training:
    - X_train: (N,7,7,1) contexts
    - y_train: (N,) target center pixels
    - X_val, y_val: same for validation
    Replace with real loader later.
    """
    N = 1000
    X = np.random.randint(0,256,size=(N,7,7,1)).astype('float32')/255.
    y = np.random.rand(N).astype('float32')
    # split
    split = int(0.8*N)
    return X[:split], y[:split], X[split:], y[split:]


def load_threshold_dataset():
    """
    Returns:
    - X: (M, F) feature matrix
    - y: (M,2) [Tn,Tp] targets
    Replace with real loader.
    """
    M, F = 200, 10
    X = np.random.rand(M, F)
    y = np.random.randint(-2,3,size=(M,2))
    return X, y


def load_region_dataset():
    """
    Returns dummy data for region classifier:
    - X_train: (P,32,32,1) patches
    - y_train: (P,) {0,1}
    - X_val, y_val: same for validation
    """
    P = 1000
    X = np.random.rand(P,32,32,1)
    y = np.random.randint(0,2,size=(P,)).astype('float32')
    split = int(0.8*P)
    return X[:split], y[:split], X[split:], y[split:]
