import tensorflow as tf
from quantum_circuit.observables import ideal_expectation_calculation
import math
from utils.activate_function import _sigmoid

def get_accuracy(circuit, obersv,input_syms,train_syms,x_samples, y_samples, value):
    values_tensor = tf.convert_to_tensor(value, dtype='float32')
    sum = len(x_samples)
    pos_num = 0.0

    for i in range(sum):
        total_circuit = circuit
        x_tensor = tf.convert_to_tensor([x_samples[i]], dtype='float32')
        perfect_outputs = ideal_expectation_calculation(total_circuit,
                                                        operators=obersv,
                                                        #                         repetitions=2000,
                                                        symbol_names= input_syms + train_syms,
                                                        symbol_values= tf.concat([x_tensor, values_tensor],axis=1))

        p_0 = _sigmoid(perfect_outputs)
        if y_samples[i] == (1 if p_0 > 0.5 else 0):  # 正例概率大于0.5 归为 正例1，否则0
            pos_num = pos_num + 1

    return pos_num / sum


def get_loss(circuit, obersv, input_syms,train_syms,x_samples, y_samples, value):
    values_tensor = tf.convert_to_tensor(value, dtype='float32')
    samples_num = len(x_samples)
    loss = 0.0

    for i in range(samples_num):
        total_circuit = circuit
        x_tensor = tf.convert_to_tensor([x_samples[i]], dtype='float32')
        perfect_outputs = ideal_expectation_calculation(total_circuit,
                                                        operators=obersv,
                                                        #                         repetitions=2000,
                                                        symbol_names=input_syms + train_syms,
                                                        symbol_values=tf.concat([x_tensor, values_tensor],axis=1))
        #         print(perfect_outputs)
        p_0 = _sigmoid(perfect_outputs)
        #         print(p_0)
        if p_0 <= 0:
            p_0 = 0
            p_0 = p_0 + 1e-5
        elif p_0 >= 1:
            p_0 = 1
            p_0 = p_0 - 1e-5
        loss = loss - (y_samples[i] * math.log(p_0) + (1 - y_samples[i]) * math.log(1 - p_0))
    #         loss = loss + (y_samples[i] - float(perfect_outputs))**2 #均方差loss
    return loss / samples_num