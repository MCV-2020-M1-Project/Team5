import numpy as np


def evaluation(A,O):

    #A =  # annotation (provided mask)
    #O =  # our background image (our mask)
    # comment: I'll assume that O and A are the masks with values 0 for black and 1 for white, and that the precision,
    # recall, and F1 measures must be computed as a % value.

    A = A/255
    O = O/255

    # TP = max(O - (1 - A), 0)            # True Positive
    # FN = max((1 - TP) - (1 - A), 0)     # False Negative
    # FP = max(O - A, 0)                  # False Positive
    # TN = max(1 - (FP + A), 0)           # True Negative
    #
    # np.sum(TP)
    # np.sum(FN)
    # np.sum(FP)
    # np.sum(TN)

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
