
import numpy as np



def categorical_cross_entropy(h, y_one_hot):
    h = np.clip(h, a_min=0.000000001, a_max=None)
    output = []
    for sample in range(len(h)):

        output.append(-1 * sum(y_one_hot[sample] * np.log(h[sample])) / len(h[sample]))

    # return the correct cross entropy!
    return output

def sigmoid(z):
    try:
        output = 1 / (1+np.e**(-z))

    except Exception as e:
        print("Sigmoid error: ")
        print(e)
        print("Max given value: " + str(np.max(z)))
        exit()

    return output

def softmax(o):
    o_new = o
    sum_val = np.sum(np.e ** o_new, axis=0, keepdims=True)

    return ((np.e ** o_new) / sum_val)

