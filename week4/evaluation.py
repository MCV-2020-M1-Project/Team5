import numpy as np


def evaluation(A,O):

    A = A/255
    O = O/255

    # Another way to do so (better in my opinion)
    A = A.reshape(-1)                   # turns into a vector
    O = O.reshape(-1)

    TP = np.dot(A, O)                   # True Positive
    FN = np.dot(A, 1-O)                 # False Negative
    FP = np.dot(1-A, O)                 # False Positive
    TN = np.dot(1-A, 1-O)               # True Negative

    # Evaluation
    P = TP / (TP + FP)                  # Precision
    R = TP / (TP + FN)                  # Recall
    F1 = 2 * (P * R / (P + R))          # F1-measure

    return P, R, F1
