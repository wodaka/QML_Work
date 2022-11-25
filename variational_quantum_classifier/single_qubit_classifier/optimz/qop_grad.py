import tensorflow as tf
from quantum_circuit.observables import ideal_expectation_calculation

def get_batch_gradient(circuit, obersv, input_syms, train_syms, x_samples, y_samples, value):
    values_tensor = tf.convert_to_tensor([value], dtype='float32')
    samples_num = len(x_samples)

    loss_grad = 0
    for i in range(samples_num):
        total_circuit = circuit
        x_tensor = tf.convert_to_tensor([x_samples[i]], dtype='float32')
        # print(x_tensor)
        # print(values_tensor)
        with tf.GradientTape() as g:
            g.watch(values_tensor)
            sampled_exp_outputs = ideal_expectation_calculation(total_circuit,
                                                                operators=obersv,
                                                                #                         repetitions=2000,
                                                                symbol_names= input_syms + train_syms,
                                                                symbol_values= tf.concat([x_tensor, values_tensor],axis=1))
        # 期望的梯度，可以求目标函数的梯度，然后执行梯度下降更新参数，最终算出的梯度经过处理成为分类结果
        sampled_finite_diff_gradients = g.gradient(sampled_exp_outputs, values_tensor)

        #for mse loss
        y_pred = (sampled_exp_outputs + 1)/2
        loss_grad = loss_grad + (y_pred - y_samples[i]) * sampled_finite_diff_gradients

    return loss_grad / samples_num