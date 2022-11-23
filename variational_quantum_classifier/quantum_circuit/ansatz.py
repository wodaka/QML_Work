import cirq
import numpy as np
from quantum_circuit.encoding import angle_encoder_layer,one_qubit_encoder_layer
from quantum_circuit.variational_circuit import one_layer_hwe,one_layer_circuit

def build_one_qubit_ansatz(qubits, input_symbols, train_symbols, layers):
    ansa = cirq.Circuit()
    encoder_layer_num = int(len(input_symbols) / 3)

    for p in range(layers):

        #对于9个特征的输出，每次取3个加入编码线路，每3个编码后 紧跟参数线路
        for i in range(encoder_layer_num):
            x_i = input_symbols[i * 3:(i + 1) * 3]
            ansa += one_qubit_encoder_layer(qubits,x_i)

            theta_index = p * encoder_layer_num + i
            theta_i = train_symbols[theta_index * 3:(theta_index + 1) * 3]
            ansa += one_layer_circuit(qubits,theta_i)
    return ansa

def build_ansatz(qubits, input_symbols, train_symbols, layers):
    ansa = cirq.Circuit()
    ansa += angle_encoder_layer(qubits, input_symbols)
    qubits_num = len(qubits)
    for p in range(layers):
        ansa += one_layer_hwe(qubits,train_symbols[qubits_num * p:qubits_num * (p+1)]) #每层参数不同
    return ansa


