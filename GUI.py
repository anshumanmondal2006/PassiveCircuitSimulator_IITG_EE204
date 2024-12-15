import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from Final_AC import CircuitSolver as ACSolver
from Final_DC import Circuit as DCSolver
matplotlib.use('qtagg')
@dataclass
class Component:
    type: str  # 'R', 'L', 'C', 'V'
    value: float
    node1: int
    node2: int
    name: str
    phase: float = 0  # Only for AC voltage sources

class CircuitDiagram:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.positions = {}
        self.labels = {}
        self.node_counter = 1

    def add_component(self, node1, node2, label):
        """Add a component between two nodes."""
        if node1 == 0:
            node1_label = "GND"
        else:
            node1_label = f"N{node1}"

        if node2 == 0:
            node2_label = "GND"
        else:
            node2_label = f"N{node2}"

        # Use a unique key for each edge to distinguish between multiple components
        edge_key = f"{label}-{len(self.graph.edges(node1_label, node2_label)) + 1}"

        self.graph.add_node(node1_label, pos=(node1, -node1))
        self.graph.add_node(node2_label, pos=(node2, -node2))
        self.graph.add_edge(node1_label, node2_label, key=edge_key, label=label)

    def validate_circuit(self):
        """Validate the circuit graph to ensure each node has at least two connections."""
        for node in self.graph.nodes:
            # Count the degree of the node, accounting for multi-edges
            if self.graph.degree[node] < 2:
                return False, f"Node {node} is not properly connected (requires at least two connections)."
        return True, ""

    def plot(self, title="Circuit Diagram"):
        """Draw the circuit diagram with multiple edges properly displayed."""
        pos = nx.spring_layout(self.graph)  # Use spring layout for better visualization
        plt.figure(figsize=(10, 6))

        # Draw the nodes and edges
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=3000,
            font_size=10,
            font_weight="bold",
        )

        # Handle edge visualization for MultiGraph
        edge_labels = {}
        for u, v, key, data in self.graph.edges(data=True, keys=True):
            label = data.get("label", "")
            if (u, v) not in edge_labels:
                edge_labels[(u, v)] = []
            edge_labels[(u, v)].append(label)

        # Draw edge labels, offset to avoid overlap
        for (u, v), labels in edge_labels.items():
            for i, label in enumerate(labels):
                nx.draw_networkx_edge_labels(
                    self.graph,
                    pos,
                    edge_labels={(u, v): label},
                    font_color="red",
                    font_size=10,
                    label_pos=0.5 + i * 0.1,  # Adjust the label position
                )

        plt.title(title)
        plt.show()


class UnifiedCircuitSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Circuit Solver")
        self.components = []
        self.mode = tk.StringVar(value="DC")
        self.frequency = tk.DoubleVar(value=0.0)
        self.setup_gui()

    def setup_gui(self):
        # Maximum time for simulation
        max_time_frame = tk.LabelFrame(self.root, text="Simulation Time (seconds)")
        max_time_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(max_time_frame, text="Max Time:").pack(side="left", padx=10)
        self.max_time = tk.DoubleVar(value=10.0)  # Default is 10 seconds
        tk.Entry(max_time_frame, textvariable=self.max_time).pack(side="left", padx=10)

        # Analysis mode selection
        mode_frame = tk.LabelFrame(self.root, text="Select Analysis Mode")
        mode_frame.pack(fill="x", padx=10, pady=5)

        tk.Radiobutton(mode_frame, text="DC", variable=self.mode, value="DC").pack(side="left", padx=10)
        tk.Radiobutton(mode_frame, text="AC", variable=self.mode, value="AC").pack(side="left", padx=10)

        # Frequency input for AC mode
        freq_label = tk.Label(mode_frame, text="Frequency (Hz):")
        freq_label.pack(side="left", padx=5)
        freq_entry = tk.Entry(mode_frame, textvariable=self.frequency, width=10)
        freq_entry.pack(side="left", padx=5)

        # Component addition section
        comp_frame = tk.LabelFrame(self.root, text="Add Component")
        comp_frame.pack(fill="x", padx=10, pady=5)

        self.comp_type = tk.StringVar(value="R")
        comp_type_label = tk.Label(comp_frame, text="Type:")
        comp_type_label.pack(side="left", padx=5)
        comp_type_menu = ttk.Combobox(comp_frame, textvariable=self.comp_type, values=["R", "L", "C", "V", "I"], width=5)
        comp_type_menu.pack(side="left", padx=5)

        self.comp_name = tk.StringVar()
        comp_name_label = tk.Label(comp_frame, text="Name:")
        comp_name_label.pack(side="left", padx=5)
        comp_name_entry = tk.Entry(comp_frame, textvariable=self.comp_name, width=10)
        comp_name_entry.pack(side="left", padx=5)

        self.comp_value = tk.DoubleVar()
        comp_value_label = tk.Label(comp_frame, text="Value:")
        comp_value_label.pack(side="left", padx=5)
        comp_value_entry = tk.Entry(comp_frame, textvariable=self.comp_value, width=10)
        comp_value_entry.pack(side="left", padx=5)

        self.node1 = tk.IntVar()
        node1_label = tk.Label(comp_frame, text="Node 1:")
        node1_label.pack(side="left", padx=5)
        node1_entry = tk.Entry(comp_frame, textvariable=self.node1, width=5)
        node1_entry.pack(side="left", padx=5)

        self.node2 = tk.IntVar()
        node2_label = tk.Label(comp_frame, text="Node 2:")
        node2_label.pack(side="left", padx=5)
        node2_entry = tk.Entry(comp_frame, textvariable=self.node2, width=5)
        node2_entry.pack(side="left", padx=5)

        self.phase = tk.DoubleVar()
        phase_label = tk.Label(comp_frame, text="Phase (°):")
        phase_label.pack(side="left", padx=5)
        phase_entry = tk.Entry(comp_frame, textvariable=self.phase, width=10)
        phase_entry.pack(side="left", padx=5)

        add_button = tk.Button(comp_frame, text="Add Component", command=self.add_component)
        add_button.pack(side="left", padx=10)

        # Components list
        self.comp_listbox = tk.Listbox(self.root, height=10)
        self.comp_listbox.pack(fill="both", padx=10, pady=5)

        delete_button = tk.Button(self.root, text="Delete Selected Component", command=self.delete_component)
        delete_button.pack(pady=5)

        # Solve buttons
        solve_button = tk.Button(self.root, text="Solve Circuit", command=self.solve_circuit)
        solve_button.pack(pady=10)

        # Preview Circuit Design button
        preview_button = tk.Button(self.root, text="Preview Circuit Design", command=self.preview_circuit_design)
        preview_button.pack(pady=10)

    def add_component(self):
        try:
            comp = Component(
                type=self.comp_type.get(),
                value=self.comp_value.get(),
                node1=self.node1.get(),
                node2=self.node2.get(),
                name=self.comp_name.get(),
                phase=np.radians(self.phase.get()) if self.mode.get() == "AC" and self.comp_type.get() in ["V", "I"] else 0
            )
            self.components.append(comp)
            self.update_component_list()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid values.")

    def delete_component(self):
        selected_index = self.comp_listbox.curselection()
        if selected_index:
            self.components.pop(selected_index[0])
            self.update_component_list()
        else:
            messagebox.showwarning("Warning", "No component selected for deletion.")

    def update_component_list(self):
        self.comp_listbox.delete(0, tk.END)
        for comp in self.components:
            phase_str = (
                f", Phase: {np.degrees(comp.phase):.1f}°"
                if comp.type in ["V", "I"] and self.mode.get() == "AC"
                else ""
            )
            self.comp_listbox.insert(
                tk.END,
                f"{comp.name} ({comp.type}): Value: {comp.value}, Nodes: {comp.node1}-{comp.node2}{phase_str}"
            )

    def solve_circuit(self):
        diagram = CircuitDiagram()
        for comp in self.components:
            label = f"{comp.type}: {comp.value}"
            diagram.add_component(comp.node1, comp.node2, label)

        valid, error_msg = diagram.validate_circuit()
        if not valid:
            messagebox.showerror("Error", f"Invalid circuit: {error_msg}")
            return
        
        max_time = self.max_time.get()
        if max_time <= 0:
            messagebox.showerror("Error", "Maximum simulation time must be positive.")
            return

        if not self.components:
            messagebox.showerror("Error", "No components to solve.")
            return

        if self.mode.get() == "AC":
            solver = ACSolver(self.frequency.get())
            solver.max_time = max_time
            for comp in self.components:
                solver.add_component(comp.type, comp.value, comp.node1, comp.node2, comp.name, comp.phase)
            voltages, currents = solver.solve()
            solver.plot_results()
            self.display_results(voltages, currents)

        elif self.mode.get() == "DC":
            solver = DCSolver()
            solver.max_time = max_time
            for comp in self.components:
                if comp.type == "R":
                    solver.add_resistor(comp.node1, comp.node2, comp.value, comp.name)
                elif comp.type == "L":
                    solver.add_inductor(comp.node1, comp.node2, comp.value, comp.name)
                elif comp.type == "C":
                    solver.add_capacitor(comp.node1, comp.node2, comp.value, comp.name)
                elif comp.type == "V":
                    solver.add_voltage_source(comp.node1, comp.node2, comp.value, comp.name)
                elif comp.type == "I":
                    solver.add_current_source(comp.node1, comp.node2, comp.value, comp.name)
            t, voltages, currents = solver.solve()
            solver.plot_results(t, voltages, currents)

    def display_results(self, voltages, currents):
        result_str = "Voltages:\n" + "\n".join(f"Node {k}: {-v}" for k, v in voltages.items()) + "\n"
        result_str += "Currents:\n" + "\n".join(f"{k}: {-v}" for k, v in currents.items())
        messagebox.showinfo("Results", result_str)

    def preview_circuit_design(self):
        """Generate and display a preview of the circuit design."""
        diagram = CircuitDiagram()
        for comp in self.components:
            label = f"{comp.type}: {comp.value}"
            if comp.type == "V" and self.mode.get() == "AC":
                label += f", Phase: {np.degrees(comp.phase):.1f}°"
            diagram.add_component(comp.node1, comp.node2, label)

        valid, error_msg = diagram.validate_circuit()
        if not valid:
            messagebox.showerror("Error", f"Invalid circuit: {error_msg}")
            return

        # Plot the diagram
        diagram.plot(title="Circuit Design Preview")


if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedCircuitSolverGUI(root)
    root.mainloop()
