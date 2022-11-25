import cirq
from quantum_circuit.variational_circuit import one_layer_circuit

def build_one_qubit_ansatz(qubits, input_syms, train_syms, layer_num):
    ansa = cirq.Circuit()
    for i in range(layer_num):
        x_i = train_syms[i * 3:(i + 1) * 3]
        ansa += one_layer_circuit(qubits, input_syms)
        ansa += one_layer_circuit(qubits,x_i)
    return ansa


