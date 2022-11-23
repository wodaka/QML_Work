import cirq
import tensorflow_quantum as tfq

ideal_expectation_calculation = tfq.layers.Expectation(
    differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01))

def z_obersv(qubits):
    pauli_Z = []
    for b in range(len(qubits)):
        pauli_Z.append(cirq.Z(qubits[b]))
    obersv = 0
    for i in range(len(qubits)):
        obersv += pauli_Z[i]
#     obersv += pauli_Z[0]
    return obersv