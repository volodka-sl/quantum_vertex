import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import COBYLA
import itertools

class QAOAVertexCover:
    def __init__(self, graph, p=1):
        self.graph = graph
        self.p = p
        self.n_vertices = len(graph.nodes())
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_quantum_circuit(self, beta, gamma):
        qr = QuantumRegister(self.n_vertices)
        cr = ClassicalRegister(self.n_vertices)
        qc = QuantumCircuit(qr, cr)
        
        # Инициализация суперпозицией
        qc.h(range(self.n_vertices))
        
        # Слои QAOA
        for layer in range(self.p):
            for edge in self.graph.edges():
                i, j = edge
                qc.rzz(2 * gamma[layer], qr[i], qr[j])
            
            for i in range(self.n_vertices):
                qc.rx(2 * beta[layer], qr[i])
        
        qc.measure(qr, cr)
        return qc
    
    def compute_cost(self, x):
        cost = sum(x)  # Кол-во вершин в покрытии
        for i, j in self.graph.edges():
            if not (x[i] or x[j]):  # Если ребро не покрыто
                cost += 1
        return cost
    
    def execute_circuit(self, beta, gamma, shots=1024):
        qc = self.create_quantum_circuit(beta, gamma)
        job = execute(qc, backend=self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        return counts
    
    def find_minimum_vertex_cover(self, shots=1024):
        optimizer = COBYLA(maxiter=100)
        
        params = np.random.uniform(0, 2*np.pi, 2*self.p)
        beta = params[:self.p]
        gamma = params[self.p:]
        
        counts = self.execute_circuit(beta, gamma, shots)

        min_cost = float('inf')
        best_solution = None
        
        for bitstring in counts:
            x = [int(bit) for bit in bitstring]
            cost = self.compute_cost(x)
            if cost < min_cost:
                min_cost = cost
                best_solution = x
                
        return best_solution, min_cost

def create_graph():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0),  # Квадрат
        (0, 2), (1, 3)  # Диагонали
    ])
    return G

def visualize_result(G, vertex_cover):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    
    nx.draw_networkx_edges(G, pos)
    
    node_colors = ['red' if vertex_cover[i] else 'lightblue' for i in range(len(G))]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Minimum Vertex Cover (red nodes)")
    plt.axis('off')
    plt.savefig('vertex_cover_result.png')
    plt.close()

def main():
    G = create_graph()
    
    qaoa = QAOAVertexCover(G, p=2)
    
    vertex_cover, cost = qaoa.find_minimum_vertex_cover()
    
    print(f"Found vertex cover: {vertex_cover}")
    print(f"Cost: {cost}")
    
    visualize_result(G, vertex_cover)

if __name__ == "__main__":
    main()
