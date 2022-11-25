import tensorflow as tf
from quantum_circuit.observables import ideal_expectation_calculation

def get_accuracy(circuit, obersv,input_syms, train_syms,x_samples, y_samples, value):
    values_tensor = tf.convert_to_tensor([value], dtype='float32')
    samples_num = len(x_samples)

    pos_num = 0.0
    for i in range(samples_num):
        total_circuit = circuit
        x_tensor = tf.convert_to_tensor([x_samples[i]], dtype='float32')
        perfect_outputs = ideal_expectation_calculation(total_circuit,
                                                        operators=obersv,
                                                        #                         repetitions=2000,
                                                        symbol_names=input_syms + train_syms,
                                                        symbol_values=tf.concat([x_tensor, values_tensor], axis=1))

        y_pred = (perfect_outputs + 1)/2
        if y_samples[i] == (1 if y_pred > 0.5 else 0):  # 正例概率大于0.5 归为 正例1，否则0
            pos_num = pos_num + 1

    return pos_num / samples_num


def get_loss(circuit, obersv, input_syms, train_syms,x_samples, y_samples, value):
    values_tensor = tf.convert_to_tensor([value], dtype='float32')
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


        #for mse loss
        y_pred = (perfect_outputs + 1)/2
        loss = loss + (y_pred - y_samples[i]) ** 2

    return loss / samples_num
