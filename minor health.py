import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
import threading
import io
import sys
import torch as nn

# --- 1. Data Simulation ---
def create_clients(n_clients=3, n_samples=1000, n_features=20, random_state=42):
    """Create a number of clients with stratified splits so each has both classes."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.5),
        n_redundant=int(n_features * 0.25),
        n_classes=2,
        random_state=random_state,
        flip_y=0.01,
        class_sep=1.0,
    )
    clients = {}
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=random_state)
    for i, (_, idx) in enumerate(skf.split(X, y), start=1):
        clients[f"client_{i}"] = {"X_train": X[idx], "y_train": y[idx]}
    return clients

# --- 2. Differential Privacy ---
def add_differential_privacy(parameters, epsilon, sensitivity=1.0, rng=None):
    """Adds Laplacian noise to parameters (coef or intercept)."""
    if epsilon <= 0:
        scale = 0.0
    else:
        scale = sensitivity / float(epsilon)
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.laplace(0.0, scale, size=parameters.shape)
    return parameters + noise

# --- 3. Federated Learning Process ---
# --- CHANGED --- Function now accepts global parameters to learn from.
def train_on_client(client_data, global_params, n_features, epsilon, rng=None):
    """
    Train on a client's data, starting from the global model state,
    and return DP-noised parameters.
    """
    X_train, y_train = client_data["X_train"], client_data["y_train"]

    # Initialize a new local model for this client
    local_model = LogisticRegression(solver='lbfgs', max_iter=500)

    # --- FIX: Part 1 ---
    # To set parameters, sklearn models must be 'fitted' at least once to
    # initialize internal attributes like shape and classes.
    dummy_X = np.zeros((2, n_features))
    dummy_y = np.array([0, 1])
    local_model.fit(dummy_X, dummy_y)

    # --- FIX: Part 2 ---
    # Now, set its parameters to the global model's state from the previous round.
    # This is the key step for iterative learning.
    local_model.coef_ = global_params[0]
    local_model.intercept_ = global_params[1]

    # Continue training (fitting) on the actual local data, improving upon the global state
    local_model.fit(X_train, y_train)

    # Add DP to both coef and intercept before returning
    private_coef = add_differential_privacy(local_model.coef_, epsilon, rng=rng)
    private_intercept = add_differential_privacy(local_model.intercept_, epsilon, rng=rng)
    return (private_coef, private_intercept)

def federated_averaging(client_models):
    """Average (coef, intercept) from all clients."""
    client_coefs, client_intercepts = zip(*client_models)
    aggregated_coefs = np.mean(client_coefs, axis=0)
    aggregated_intercepts = np.mean(client_intercepts, axis=0)
    return (aggregated_coefs, aggregated_intercepts)

# --- Thread-safe logger helper ---
class TkTextLogger:
    """File-like object that appends to a Tk Text widget safely from any thread."""
    def __init__(self, root, text_widget):
        self.root = root
        self.text_widget = text_widget

    def write(self, s):
        if not s:
            return
        self.root.after(0, self.text_widget.insert, tk.END, s)
        self.root.after(0, self.text_widget.see, tk.END)

    def flush(self):
        pass

# --- 4. Main Application Class ---
class FederatedLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Federated Learning Simulation")
        self.root.geometry("900x640")

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Label(controls_frame, text="Rounds:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.rounds_var = tk.StringVar(value="10")
        ttk.Entry(controls_frame, textvariable=self.rounds_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(controls_frame, text="Clients:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.clients_var = tk.StringVar(value="5")
        ttk.Entry(controls_frame, textvariable=self.clients_var, width=10).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(controls_frame, text="Epsilon:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.epsilon_var = tk.StringVar(value="1.0")
        ttk.Entry(controls_frame, textvariable=self.epsilon_var, width=10).grid(row=0, column=5, padx=5, pady=5)

        self.start_button = ttk.Button(controls_frame, text="Start Simulation", command=self.start_simulation_thread)
        self.start_button.grid(row=0, column=6, padx=20, pady=5)

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        log_frame = ttk.LabelFrame(output_frame, text="Logs", padding="10")
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=60, height=10)
        self.log_area.pack(fill=tk.BOTH, expand=True)

        graph_frame = ttk.LabelFrame(output_frame, text="Accuracy Plot", padding="10")
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.fig, self.ax = plt.subplots(figsize=(5.5, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_simulation_thread(self):
        self.start_button.config(state=tk.DISABLED)
        self.log_area.delete('1.0', tk.END)
        self.ax.clear()
        self.ax.set_title('Federated Learning Model Accuracy')
        self.ax.set_xlabel('Communication Round')
        self.ax.set_ylabel('Accuracy')
        self.ax.grid(True)
        self.canvas.draw()

        thread = threading.Thread(target=self.run_simulation, daemon=True)
        thread.start()

    def run_simulation(self):
        old_stdout = sys.stdout
        sys.stdout = TkTextLogger(self.root, self.log_area)

        try:
            n_rounds = int(self.rounds_var.get())
            n_clients = int(self.clients_var.get())
            epsilon = float(self.epsilon_var.get())
            n_features = 20
            rng = np.random.default_rng(42)

            print("Starting Federated Learning Simulation...")
            print(f"Number of rounds: {n_rounds}")
            print(f"Number of clients: {n_clients}")
            print(f"Differential Privacy Epsilon: {epsilon}\n")

            clients = create_clients(n_clients=n_clients, n_samples=1000, n_features=n_features, random_state=42)
            X_test, y_test = make_classification(
                n_samples=400, n_features=n_features,
                n_informative=int(n_features * 0.5),
                n_redundant=int(n_features * 0.25),
                n_classes=2, random_state=43, flip_y=0.01, class_sep=1.0
            )
            
            # This global model is now used mainly for evaluation.
            # The parameters are stored in the `global_model_params` tuple.
            global_model = LogisticRegression(solver='lbfgs')
            dummy_X = np.zeros((2, n_features))
            dummy_y = np.array([0, 1])
            global_model.fit(dummy_X, dummy_y)

            # Start with zeroed parameters. This is where the learning state is stored.
            global_model_params = (np.zeros_like(global_model.coef_), np.zeros_like(global_model.intercept_))
            accuracy_history = []

            for round_num in range(1, n_rounds + 1):
                print(f"--- Round {round_num} ---")
                client_models = []

                for client_id, client_data in clients.items():
                    print(f"Training on {client_id}...")
                    # --- CHANGED --- Pass the global parameters to each client
                    local_model_params = train_on_client(
                        client_data, global_model_params, n_features, epsilon, rng=rng
                    )
                    client_models.append(local_model_params)

                # Aggregate
                global_model_params = federated_averaging(client_models)
                print("Model aggregation complete.")

                # Evaluate
                global_model.coef_ = global_model_params[0]
                global_model.intercept_ = global_model_params[1]
                y_pred = global_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_history.append(accuracy)
                print(f"Global model accuracy: {accuracy:.4f}\n")

                # Update plot in the main thread
                self.root.after(0, self.update_plot, round_num, accuracy_history, n_rounds)

            print("Simulation finished.")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            sys.stdout = old_stdout
            self.root.after(0, self.enable_button)

    def update_plot(self, round_num, accuracy_history, n_rounds):
        self.ax.clear()
        self.ax.plot(range(1, round_num + 1), accuracy_history, marker='o', linestyle='-')
        self.ax.set_title('Federated Learning Model Accuracy')
        self.ax.set_xlabel('Communication Round')
        self.ax.set_ylabel('Accuracy')
        # Ensure x-axis ticks are integers
        self.ax.set_xticks(np.arange(1, n_rounds + 1, step=max(1, n_rounds // 10)))
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True)
        self.canvas.draw()

    def enable_button(self):
        self.start_button.config(state=tk.NORMAL)

if __name__ == '__main__':
    root = tk.Tk()
    app = FederatedLearningApp(root)
    root.mainloop()