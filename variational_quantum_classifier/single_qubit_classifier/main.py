import argparse
import time
import cirq
import tensorflow_quantum as tfq
import sympy
from cirq.contrib.svg import SVGCircuit
import copy
import math
import numpy as np
import random


from quantum_circuit.ansatz import build_one_qubit_ansatz
from optimz.optim import Adam
from optimz.qop_grad import get_batch_gradient
from quantum_circuit.observables import z_obersv
from metric.metrics import get_accuracy,get_loss
from dataset.data_preprocessing import circle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser("VQC")
parser.add_argument('--learning_rate', type=float, default=0.6, help='learning rate')
parser.add_argument('--report_freq', type=int, default=10, help='report frequency')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# circuit
parser.add_argument('--n_qubits', type=int, default=1, help='number of qubits')
parser.add_argument('--n_layers', type=int, default=3, help='number of layers')

args = parser.parse_args()

ideal_expectation_calculation = tfq.layers.Expectation(
    differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01))

def get_validate_sets(x_samples, y_samples):
    val_index = []
    num = math.ceil(len(x_samples) * 0.6)
    for i in range(len(x_samples)):
        val_index.append(i)
    random.shuffle(val_index)
    validate_index = val_index[0:num]
    x_validate = []
    y_validate = []
    for i in range(len(validate_index)):
        x_validate.append(x_samples[validate_index[i]])
        y_validate.append(y_samples[validate_index[i]])
    return x_validate, y_validate


def main():
    # x_train,y_train,x_test,y_test = get_mnist()
    #读取训练数据 两维数据扩充为3维
    num_training = 200
    num_test = 2000

    Xdata, y_train = circle(num_training)
    x_train = np.hstack((Xdata, np.zeros((Xdata.shape[0], 1))))

    Xtest, y_test = circle(num_test)
    x_test = np.hstack((Xtest, np.zeros((Xtest.shape[0], 1))))

    #搭建量子线路
    qubits_num = args.n_qubits
    layer_num = args.n_layers

    #construct quantum circuit
    my_circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, qubits_num)
    obersv = z_obersv(qubits)

    #实际输入的参数只有一个，ad_syms = input_repeat * train_symbols_1 + train_symbols_2 #紧凑型
    #最初的方案：一层编码一层训练
    #define input_variable & train_variable for one_qubit circuit  U(theta + x * w)
    input_syms = sympy.symbols('x0:%d' % 3)
    param_num_train = len(input_syms) * layer_num
    train_syms = sympy.symbols('t0:%d' % param_num_train)

    params_order = []
    for i in range(len(train_syms)):
        params_order.append(i)

    #add encode circuit and variational circuit
    ansatz = build_one_qubit_ansatz(qubits, input_syms, train_syms, layer_num)
    # my_circuit = my_circuit + cirq.H.on(qubits[0])
    my_circuit = my_circuit + ansatz
    # SVGCircuit(my_circuit)

    value = np.random.uniform(-2 * np.pi, 2 * np.pi, param_num_train).tolist()

    epoch = args.epochs
    batch_size = args.batch_size
    update_num = math.ceil(len(x_train) / batch_size)
    adam_optimizer = Adam(lr=args.learning_rate)
    for e in range(epoch):
        print('epoch %d' % (e))
        time_start = time.time()
        if e == e:
            for i in range(update_num):
                #     for i in range(1):
                if i != (update_num - 1):
                    x_samples = x_train[i * batch_size:(i + 1) * batch_size]
                    y_samples = y_train[i * batch_size:(i + 1) * batch_size]
                else:
                    x_samples = x_train[i * batch_size:len(x_train)]
                    y_samples = y_train[i * batch_size:len(y_train)]

                # 将冻结的value的gradient置0
                loss_grad = get_batch_gradient(my_circuit,obersv,input_syms, train_syms, x_samples, y_samples, value)

                value_update = copy.deepcopy(value)

                params_update = dict(zip(params_order, value_update))

                adam_optimizer.update(params_update, loss_grad[0])

                for key in params_update.keys():
                    value_update[key] = params_update[key]

                value = np.clip(value_update, -np.pi * 2, np.pi * 2).tolist()

                if i%args.report_freq == 0:
                    loss = get_loss(my_circuit,obersv,input_syms, train_syms,x_train,y_train,value)
                    train_acc = get_accuracy(my_circuit,obersv,input_syms, train_syms,x_train,y_train,value)
                    test_acc = get_accuracy(my_circuit,obersv,input_syms, train_syms,x_test,y_test,value)
                    time_end = time.time()
                    print("update_%.d  loss: %.4f train_acc: %.4f test_acc: %.4f time: %.4f" %(i, loss,train_acc,test_acc,time_end-time_start))

if __name__ == '__main__':
    main()