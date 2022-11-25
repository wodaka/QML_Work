import cirq

#input_symbols = [x1,x2,x3]
#output = Rz(x1)Ry(x2)Rz(x3)
def one_qubit_encoder_layer(qubits, input_symbols):
    el_circuit = cirq.Circuit()
    el_circuit += cirq.rz(input_symbols[0]).on(qubits[0])
    el_circuit += cirq.ry(input_symbols[1]).on(qubits[0])
    el_circuit += cirq.rz(input_symbols[2]).on(qubits[0])
    return el_circuit

def angle_encoder_layer(qubits, input_symbols):
    el_circuit = cirq.Circuit()
    for i in range(len(qubits)):
        el_circuit += cirq.ry(input_symbols[i]).on(qubits[i])
    return el_circuit