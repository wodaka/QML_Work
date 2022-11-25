import cirq

def one_layer_circuit(qubits,train_symbols):
    #creat a quantum circuit
    ol_circuit = cirq.Circuit()
    ol_circuit += cirq.rz(train_symbols[0]).on(qubits[0])
    ol_circuit += cirq.ry(train_symbols[1]).on(qubits[0])
    ol_circuit += cirq.rz(train_symbols[2]).on(qubits[0])
    return ol_circuit