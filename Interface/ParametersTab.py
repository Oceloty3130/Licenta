import os
import json
import torch
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLineEdit, QSpinBox,
    QDoubleSpinBox, QPushButton, QFileDialog, QHBoxLayout,
    QFormLayout, QSizePolicy, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot
from Neural_Network.Lyapunov_Model.security import (
    verify_model_integrity_encrypted,
    load_model
)
from Neural_Network.Lyapunov_Model.network import LyapunovNet

class ParametersTab(QWidget):
    training_requested = Signal(dict)

    def __init__(self, minimum_graph_size, maximum_graph_size, parent=None):
        super().__init__(parent)
        self.setMinimumSize(minimum_graph_size)
        self.setMaximumSize(maximum_graph_size)

        # State flags
        self.is_training = False
        self._loaded = False
        self._save_path = None       # base directory for models

        layout = QVBoxLayout(self)

        # — System Order —
        self.order_group_box = QGroupBox("System Order")
        order_layout = QFormLayout()
        self.system_order_input = QSpinBox()
        self.system_order_input.setRange(1, 10)
        self.system_order_input.setValue(1)
        self.system_order_input.valueChanged.connect(self.update_equations)
        order_layout.addRow("Select System Order:", self.system_order_input)
        self.order_group_box.setLayout(order_layout)
        layout.addWidget(self.order_group_box)

        # — Equations —
        self.equations_group_box = QGroupBox("Equations")
        self.equations_layout = QVBoxLayout()
        self.equation_inputs = []
        self.update_equations()
        self.equations_group_box.setLayout(self.equations_layout)
        layout.addWidget(self.equations_group_box)

        # — Hyperparameters —
        self.hyperparameters_group_box = QGroupBox("Hyperparameters")
        hyper_layout = QFormLayout()
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setDecimals(5)
        self.learning_rate_input.setValue(0.01)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(100)
        self.hidden_layer_size_input = QSpinBox()
        self.hidden_layer_size_input.setRange(1, 1000)
        self.hidden_layer_size_input.setValue(10)
        self.alpha_input = QDoubleSpinBox()
        self.alpha_input.setRange(0.0001, 1.0)
        self.alpha_input.setDecimals(5)
        self.alpha_input.setValue(0.01)
        self.step_input = QDoubleSpinBox()
        self.step_input.setRange(0.0001, 10.0)
        self.step_input.setDecimals(5)
        self.step_input.setValue(0.01)
        hyper_layout.addRow("Learning Rate:", self.learning_rate_input)
        hyper_layout.addRow("Epochs:", self.epochs_input)
        hyper_layout.addRow("Hidden Layer Size:", self.hidden_layer_size_input)
        hyper_layout.addRow("Alpha:", self.alpha_input)
        hyper_layout.addRow("Step:", self.step_input)
        self.hyperparameters_group_box.setLayout(hyper_layout)
        layout.addWidget(self.hyperparameters_group_box)

        # — Domain —
        self.domain_group_box = QGroupBox("Domain")
        domain_layout = QFormLayout()
        self.x_min_input = QDoubleSpinBox()
        self.x_min_input.setRange(-1000.0, 1000.0)
        self.x_min_input.setValue(-1.0)
        self.x_max_input = QDoubleSpinBox()
        self.x_max_input.setRange(-1000.0, 1000.0)
        self.x_max_input.setValue(1.0)
        self.x_min_input.valueChanged.connect(self.validate_domain)
        domain_layout.addRow("X Min:", self.x_min_input)
        domain_layout.addRow("X Max:", self.x_max_input)
        self.domain_group_box.setLayout(domain_layout)
        layout.addWidget(self.domain_group_box)

        # — Buttons: Base Path, Load, New, Start, Stop —
        btn_layout = QHBoxLayout()
        self.file_path_button = QPushButton("Select Base Path")
        self.file_path_button.clicked.connect(self.select_save_path)
        btn_layout.addWidget(self.file_path_button)
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model_params)
        btn_layout.addWidget(self.load_button)
        self.new_button = QPushButton("New Model")
        self.new_button.clicked.connect(self.new_model)
        btn_layout.addWidget(self.new_button)
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        btn_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.stop_button)
        layout.addLayout(btn_layout)

        # — Status Label —
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        layout.addStretch()

    @Slot()
    def select_save_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Base Directory for Models")
        if path:
            self._save_path = path
            self.status_label.setText(f"Base path set: {path}")

    @Slot()
    def load_model_params(self):
        # Must have chosen a base path first
        if not self._save_path:
            QMessageBox.warning(self, "Load Error", "Set base path first!")
            return

        order = self.system_order_input.value()
        hidden = self.hidden_layer_size_input.value()
        model_dir = os.path.join(self._save_path, f"model_{order}_{hidden}")
        if not os.path.isdir(model_dir):
            QMessageBox.warning(
                self, "Load Error",
                f"Model folder not found:\n{model_dir}"
            )
            return

        # Verify the security bundle under Model_Parameters
        sec_dir = os.path.join(model_dir, "Model_Parameters")
        valid = verify_model_integrity_encrypted(sec_dir, model_dir)
        if valid:
            # instanțiezi LyapunovNet și îl încarci din Model_Parameters
            device = torch.device("cpu")
            net = LyapunovNet(order, hidden, alpha=0.01).to(device)
            net = load_model(model_dir, net, device, order, hidden, alpha=0.01)
            QMessageBox.information(self, "Load OK", "Model încărcat cu succes!")

        # Now load params.json
        params_file = os.path.join(model_dir, "params.json")
        try:
            with open(params_file, "r") as f:
                params = json.load(f)
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Could not read params.json:\n{e}"
            )
            return

        # Ensure the model matches the chosen order/hidden size
        if params.get("order") != order or params.get("hidden_layer_size") != hidden:
            QMessageBox.warning(
                self, "Incompatible Model",
                "Loaded model parameters do not match current settings."
            )
            return

        # Populate UI fields
        self.system_order_input.setValue(params["order"])
        self.learning_rate_input.setValue(params.get("learning_rate", 0.01))
        self.epochs_input.setValue(params.get("max_epochs", 100))
        self.hidden_layer_size_input.setValue(params["hidden_layer_size"])
        self.alpha_input.setValue(params.get("alpha", 0.01))
        self.step_input.setValue(params.get("step", 0.01))
        self.x_min_input.setValue(params.get("x_min", -1.0))
        self.x_max_input.setValue(params.get("x_max", 1.0))

        exprs = params.get("expressions", [])
        self.update_equations()
        for inp, e in zip(self.equation_inputs, exprs):
            inp.setText(e)

        # Mark as loaded
        self._loaded = True
        self.load_button.setEnabled(False)
        self.new_button.setEnabled(True)
        self.status_label.setText(f"Model loaded from:\n{model_dir}")

    @Slot()
    def new_model(self):
        self._loaded = False
        self.load_button.setEnabled(True)
        self.system_order_input.setEnabled(True)
        self.hidden_layer_size_input.setEnabled(True)
        self.update_equations()
        self.status_label.setText("Ready: define new model parameters.")

    @Slot()
    def start_training(self):
        if not self._save_path:
            QMessageBox.warning(self, "Start Error", "Select base path first!")
            return
        exprs = [w.text().strip() for w in self.equation_inputs]
        if any(not e for e in exprs):
            QMessageBox.warning(self, "Input Error", "Define all equations!")
            return
        params = {
            'order': self.system_order_input.value(),
            'learning_rate': self.learning_rate_input.value(),
            'max_epochs': self.epochs_input.value(),
            'hidden_layer_size': self.hidden_layer_size_input.value(),
            'alpha': self.alpha_input.value(),
            'step': self.step_input.value(),
            'x_min': self.x_min_input.value(),
            'x_max': self.x_max_input.value(),
            'expressions': exprs
        }
        model_dir = os.path.join(self._save_path,
                                 f"model_{params['order']}_{params['hidden_layer_size']}")
        os.makedirs(model_dir, exist_ok=True)
        # Write params.json if new or loaded
        with open(os.path.join(model_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)
        self.is_training = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.load_button.setEnabled(False)
        self.new_button.setEnabled(False)
        self.system_order_input.setEnabled(False)
        self.hidden_layer_size_input.setEnabled(False)
        self.alpha_input.setEnabled(False)
        self.step_input.setEnabled(False)
        self.x_min_input.setEnabled(False)
        self.x_max_input.setEnabled(False)
        self.learning_rate_input.setEnabled(False)
        self.epochs_input.setEnabled(False)
        for inp in self.equation_inputs:
            inp.setEnabled(False)
        self.status_label.setText("Training started...")
        self.training_requested.emit({'params': params, 'model_dir': model_dir})

    @Slot()
    def stop_training(self):
        self.is_training = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.new_button.setEnabled(True)
        self.system_order_input.setEnabled(True)
        self.hidden_layer_size_input.setEnabled(True)
        self.alpha_input.setEnabled(True)
        self.step_input.setEnabled(True)
        self.x_min_input.setEnabled(True)
        self.x_max_input.setEnabled(True)
        self.learning_rate_input.setEnabled(True)
        self.epochs_input.setEnabled(True)
        for inp in self.equation_inputs:
            inp.setEnabled(False)
        self.status_label.setText("Training stopped.")

    def update_equations(self):
        order = self.system_order_input.value()
        for w in self.equation_inputs:
            w.deleteLater()
        self.equation_inputs.clear()
        for i in range(order):
            eq = QLineEdit()
            eq.setPlaceholderText(f"dx{i+1}/dt = ...")
            eq.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.equations_layout.addWidget(eq)
            self.equation_inputs.append(eq)

    def validate_domain(self):
        if self.x_min_input.value() > self.x_max_input.value():
            self.x_min_input.setValue(self.x_max_input.value())
