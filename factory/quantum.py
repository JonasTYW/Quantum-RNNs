import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import pennylane as qml

class QGRU(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=6,
                n_qlayers=1,
                n_vrotations=3,
                batch_first=False,
                return_sequences=False, 
                return_state=False,
                backend="default.qubit"
                ):
        super(QGRU, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        #gates
        
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_reset = [f"wire_reset_{i}" for i in range(self.n_qubits)]
        self.wires_memory = [f"wire_memory_{i}" for i in range(self.n_qubits)]

        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_reset = qml.device(self.backend, wires=self.wires_reset)
        self.dev_memory = qml.device(self.backend, wires=self.wires_memory)
        
        def ansatz(params, wires_type):
            # Entangling layer.
            for i in range(1,3): 
                for j in range(self.n_qubits):
                    if j + i < self.n_qubits:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i]])
                    else:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i - self.n_qubits]])

            # Variational layer.
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])
                
        def VQC(features, weights, wires_type):
            # Preproccess input data to encode the initial state.
            #qml.templates.AngleEmbedding(features, wires=wires_type)
            ry_params = [torch.arctan(feature) for feature in features]
            rz_params = [torch.arctan(feature**2) for feature in features]
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i], wires=wires_type[i])
                qml.RZ(rz_params[i], wires=wires_type[i])
        
            #Variational block.
            qml.layer(ansatz, self.n_qlayers, weights, wires_type = wires_type)

        def _circuit_update(inputs, weights):
            VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch", diff_method='adjoint')

        def _circuit_reset(inputs, weights):
            VQC(inputs, weights, self.wires_reset)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_reset]
        self.qlayer_reset = qml.QNode(_circuit_reset, self.dev_reset, interface="torch", diff_method='adjoint')
        
        def _circuit_memory(inputs, weights):
            VQC(inputs, weights, self.wires_memory)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_memory]
        self.qlayer_memory = qml.QNode(_circuit_memory, self.dev_memory, interface="torch", diff_method='adjoint')

        weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_vrotations, n_qubits) = ({self.n_qlayers}, {self.n_vrotations}, {self.n_qubits})")

        self.h2h = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        self.clayer_in2 = torch.nn.Linear(self.concat_size, self.n_qubits)
        self.VQC = {
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'reset': qml.qnn.TorchLayer(self.qlayer_reset, weight_shapes),
            'memory': qml.qnn.TorchLayer(self.qlayer_memory, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        self.clayer_out2 = torch.nn.Linear(self.n_qubits, self.hidden_size)
        self.clayer_out3 = torch.nn.Linear(self.n_qubits, self.hidden_size)
        #self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state 
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t = init_states
            h_t = h_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            if self.batch_first is True:
                x_t = x[:, t , :]
            else:
                x_t = x[t, : , :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            z_t = torch.sigmoid(self.clayer_out(self.VQC['update'](y_t)))  # update block
            r_t = torch.sigmoid(self.clayer_out2(self.VQC['reset'](y_t)))  # reset block
            
            h_t_new = self.h2h(h_t)
            
            g_t = (r_t * h_t)
            
            v_prime_t = torch.cat((g_t, x_t), dim=1)
            
            y_prime_t = self.clayer_in2(v_prime_t)
            
            h_prime_t = torch.tanh(self.clayer_out3(self.VQC['memory'](y_prime_t))) # memory
            
            o1_t = (z_t * h_t)
            o2_t = ((-1)*(z_t - 1)) * h_prime_t
            
            h_t = o1_t + o2_t

            h_t = h_t.unsqueeze(0)
            hidden_seq.append(h_t)
        hidden_seq = torch.cat(hidden_seq, dim=0)
        #hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, h_t