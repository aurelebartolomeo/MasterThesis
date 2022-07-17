import numpy as np
from scipy.stats import norm, multivariate_normal, bernoulli
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def generate_data(n, alpha0, alpha1A, alpha1B, time_steps):
    b_0, b_1 = multivariate_normal.rvs(mean=np.zeros(2), cov=np.array([[1, 0.5],[0.5, 1]]), size=n).transpose()
    t = np.arange(time_steps)
    epsilon = norm.rvs(loc=0, scale=1, size=n*time_steps)
    
    Y_A = []
    Y_B = []
    for i in range(n):
        if(i < n/2):
            new_individual = alpha0 + b_0[i] + (alpha1A + b_1[i]) * t + epsilon[time_steps*i : time_steps*i + time_steps]
            Y_A.append(new_individual)
        else: 
            new_individual = alpha0 + b_0[i] + (alpha1B + b_1[i]) * t + epsilon[time_steps*i : time_steps*i + time_steps]
            Y_B.append(new_individual)
    
    data = Y_A + Y_B
    labels = np.append(np.ones(n//2), np.ones(n//2)*0) #A:1, B:0
    return (data, labels)

def generate_data_MCAR(data, p):
    data_ = np.array(data[0])
    labels = data[1]
    time_steps = len(data_[0])
    n = len(data_)
    miss_positions = np.reshape(bernoulli.rvs(p, size=n*time_steps), (n, time_steps))
    data_[miss_positions==1] = None
    return (data_, labels)


def generate_data_MAR(data, p_A, p_B):
    data_ = np.array(data[0])
    labels = data[1]
    
    dataA = data_[labels==1]
    time_stepsA = len(dataA[0])
    nA = len(dataA)
    miss_positionsA = np.reshape(bernoulli.rvs(p_A, size=nA*time_stepsA), (nA, time_stepsA))
    dataA[miss_positionsA==1] = None
    labelsA = np.ones(nA)
    
    dataB = data_[labels==0]
    time_stepsB = len(dataB[0])
    nB = len(dataB)
    miss_positionsB = np.reshape(bernoulli.rvs(p_B, size=nB*time_stepsB), (nB, time_stepsB))
    dataB[miss_positionsB==1] = None
    labelsB = np.zeros(nB)

    data_ = np.append(dataA, dataB)
    labels = np.append(labelsA, labelsB)

    return (data_, labels)
    
def generate_data_MNAR(data, p_A, p_B, treshold):    
    data_ = np.array(data[0])
    labels = data[1]
    
    dataA = np.array(data_[labels==1])
    time_stepsA = len(dataA[0])
    nA = len(dataA)
    miss_positionsA = np.reshape(bernoulli.rvs(p_A, size=nA*time_stepsA), (nA, time_stepsA))
    dataA[miss_positionsA==1] = None
    dataA[dataA >= treshold] = None
    labelsA = np.ones(nA)
    
    dataB = data_[labels==0]
    time_stepsB = len(dataB[0])
    nB = len(dataB)
    miss_positionsB = np.reshape(bernoulli.rvs(p_B, size=nB*time_stepsB), (nB, time_stepsB))
    dataB[miss_positionsB==1] = None
    dataA[dataB >= treshold] = None
    labelsB = np.zeros(nB)

    data_ = np.append(dataA, dataB)
    labels = np.append(labelsA, labelsB)

    return (data_, labels)

def CCA(data):
    data_ = data[0]
    labels = data[1]
    dataCCA = [l for l in data_ if not np.isnan(l).any()]
    labelsCCA = [labels[i] for i in range(len(data_)) if not np.isnan(data_[i]).any()]
    return (dataCCA, labelsCCA)

def LOCF(data):
    pass

def mean_imputation(data):
    pass

def regression_imputation(data):
    pass

def multiple_imputation(data):            
    pass

def RNN_MSE(data):
    data_ = data[0]
    labels = data[1]
    X_train, X_test, y_train, y_test = train_test_split(data_, labels, test_size=0.20, random_state=42)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = keras.models.Sequential()

    model.add(layers.SimpleRNN(32, input_shape=(10,1)))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, epochs=10)

    score = model.evaluate(X_test, y_test, verbose = 0) 
    return score[1]


