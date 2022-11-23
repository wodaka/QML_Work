import argparse
import time
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy
from cirq.contrib.svg import SVGCircuit
import copy
import math
import numpy as np
import random

from dataset.data_preprocessing import get_mnist
from quantum_circuit.ansatz import build_ansatz
from optimz.optim import Adam
from optimz.qop_grad import get_batch_gradient
from quantum_circuit.observables import z_obersv
from metric.metrics import get_accuracy,get_loss

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser("VQC")
parser.add_argument('--data', type=str, default='./dataset/mnist.npz', help='location of the data corpus')
parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
parser.add_argument('--report_freq', type=int, default=10, help='report frequency')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
# circuit
parser.add_argument('--n_qubits', type=int, default=9, help='number of qubits')
parser.add_argument('--n_encode_layers', type=int, default=1, help='number of encoder layers')
parser.add_argument('--n_layers', type=int, default=2, help='number of layers')

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
    x_train,y_train,x_test,y_test = get_mnist()

    qubits_num = args.n_qubits
    layer_num = args.n_layers
    # param_num = qubits_num  * layer_num * 3 * 3 #one_qubit_encoding
    # x_len = 9

    param_num = qubits_num  * layer_num
    #construct quantum circuit
    my_circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, qubits_num)
    obersv = z_obersv(qubits)

    #define input_variable & train_variable
    input_syms = sympy.symbols('x0:%d'%qubits_num)
    train_syms = sympy.symbols('t0:%d'%param_num)

    params_order = []
    for i in range(len(train_syms)):
        params_order.append(i)

    #add encode circuit and variational circuit
    ansatz = build_ansatz(qubits, input_syms, train_syms, layer_num)
    # my_circuit = my_circuit + cirq.H.on(qubits[0])
    my_circuit = my_circuit + ansatz
    # SVGCircuit(my_circuit)

    epoch = args.epochs
    batch_size = args.batch_size
    train_test = 2000
    sample_test = 800

    x_train = x_train[0:train_test]
    y_train = y_train[0:train_test]

    #测试数据集 选一部分作为验证
    x_test = x_test[0:math.ceil(sample_test*0.8)]
    y_test = y_test[0:math.ceil(sample_test*0.8)]

    x_validate,y_validate = get_validate_sets(x_train,y_train)
    xt_validate,yt_validate = get_validate_sets(x_test,y_test)

    update_num = math.ceil(len(x_train) / batch_size)

    adam_optimizer = Adam(lr=args.learning_rate)

    # value = [(np.ones(param_num) * np.pi/4).tolist()]
    value = [np.random.uniform(-2 * np.pi, 2 * np.pi, param_num).tolist()]
    value_init = copy.deepcopy(value)

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
                loss_grad = get_batch_gradient(my_circuit,obersv,input_syms,train_syms,x_samples, y_samples, value)

                value_update = copy.deepcopy(value)

                params_update = dict(zip(params_order, value_update))

                adam_optimizer.update(params_update, loss_grad)

                for key in params_update.keys():
                    value_update[key] = params_update[key]

                value = np.clip(value_update, -np.pi * 2, np.pi * 2).tolist()
            #         value = np.clip(value_update,0,np.pi*4).tolist()

                if i%args.report_freq == 0:
                    loss = get_loss(my_circuit,obersv,input_syms,train_syms,x_validate,y_validate,value)
                    train_acc = get_accuracy(my_circuit,obersv,input_syms,train_syms,x_validate,y_validate,value)
                    test_acc = get_accuracy(my_circuit,obersv,input_syms,train_syms,xt_validate,yt_validate,value)
                    time_end = time.time()
                    print("update_%.d  loss: %.4f train_acc: %.4f test_acc: %.4f time: %.4f" %(i, loss,train_acc,test_acc,time_end-time_start))

if __name__ == '__main__':
    main()