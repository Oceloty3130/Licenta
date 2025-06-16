from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, \
    QPushButton, QFileDialog, QHBoxLayout, QFormLayout, QSizePolicy

class ParametersTab(QWidget):
    def __init__(self, minimum_graph_size, maximum_graph_size, parent=None):
        super().__init__(parent)

        self.setMinimumSize(minimum_graph_size)
        self.setMaximumSize(maximum_graph_size)

        self.is_training = False  # Variabila pentru a urmări starea antrenării

        # Layout principal
        layout = QVBoxLayout(self)

        # Secțiune pentru selectarea ordinului sistemului
        self.order_group_box = QGroupBox("System Order", self)
        self.order_layout = QFormLayout()
        self.system_order_input = QSpinBox(self)
        self.system_order_input.setRange(1, 10)  # Sistem de ordin 1-10
        self.system_order_input.setValue(1)  # Valoarea implicită este 1

        # Când se schimbă valoarea, actualizăm ecuațiile
        self.system_order_input.valueChanged.connect(self.update_equations)

        self.order_layout.addRow("Select System Order:", self.system_order_input)
        self.order_group_box.setLayout(self.order_layout)

        # Setăm dimensiune flexibilă pentru grupul "System Order"
        self.order_group_box.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        # Secțiune pentru ecuații
        self.equations_group_box = QGroupBox("Equations", self)
        self.equations_layout = QVBoxLayout()

        # Listă pentru stocarea câmpurilor de ecuații
        self.equation_inputs = []

        # Adăugăm câte ecuații sunt necesare pe baza ordinului
        self.update_equations()

        # Setăm un QSizePolicy pentru grupul de ecuații pentru a fi mai flexibil
        self.equations_group_box.setLayout(self.equations_layout)
        self.equations_group_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Secțiunea pentru Hyperparameters
        self.hyperparameters_group_box = QGroupBox("Hyperparameters", self)
        hyperparameters_layout = QFormLayout()

        # Parametrii Hyperparameters cu Range (pentru a controla dimensiunea inputurilor)
        self.learning_rate_input = QDoubleSpinBox(self)
        self.learning_rate_input.setRange(0.0001, 1.0)
        self.learning_rate_input.setValue(0.01)
        self.learning_rate_input.setDecimals(5)

        self.input_size_input = QSpinBox(self)
        self.input_size_input.setRange(1, 1000)
        self.input_size_input.setValue(64)

        self.hidden_layer_size_input = QSpinBox(self)
        self.hidden_layer_size_input.setRange(1, 1000)
        self.hidden_layer_size_input.setValue(128)

        self.alpha_input = QDoubleSpinBox(self)
        self.alpha_input.setRange(0.0001, 1.0)
        self.alpha_input.setValue(0.01)

        self.step_input = QDoubleSpinBox(self)
        self.step_input.setRange(0.0001, 10.0)
        self.step_input.setValue(0.01)

        # Adăugăm la layout-ul pentru Hyperparameters
        hyperparameters_layout.addRow("Learning Rate:", self.learning_rate_input)
        hyperparameters_layout.addRow("Input Size:", self.input_size_input)
        hyperparameters_layout.addRow("Hidden Layer Size:", self.hidden_layer_size_input)
        hyperparameters_layout.addRow("Alpha:", self.alpha_input)
        hyperparameters_layout.addRow("Step:", self.step_input)

        self.hyperparameters_group_box.setLayout(hyperparameters_layout)

        # Secțiunea pentru Domeniu
        self.domain_group_box = QGroupBox("Domain", self)
        domain_layout = QFormLayout()

        self.x_min_input = QDoubleSpinBox(self)
        self.x_min_input.setRange(-1000.0, 1000.0)
        self.x_min_input.setValue(-10.0)

        self.x_max_input = QDoubleSpinBox(self)
        self.x_max_input.setRange(-1000.0, 1000.0)
        self.x_max_input.setValue(10.0)

        # Validăm că x_min <= x_max
        self.x_min_input.valueChanged.connect(self.validate_domain)

        # Adăugăm la layout-ul pentru Domeniu
        domain_layout.addRow("X Min:", self.x_min_input)
        domain_layout.addRow("X Max:", self.x_max_input)

        self.domain_group_box.setLayout(domain_layout)

        # Buton pentru selectarea căii de salvare
        self.file_path_button = QPushButton("Select Save Path", self)
        self.file_path_button.clicked.connect(self.select_save_path)

        # Buton pentru Start Training
        self.start_button = QPushButton("Start Training", self)
        self.start_button.clicked.connect(self.start_training)

        # Buton pentru Stop Training
        self.stop_button = QPushButton("Stop Training", self)
        self.stop_button.setEnabled(False)  # La început, butonul "Stop" este dezactivat
        self.stop_button.clicked.connect(self.stop_training)

        # Adăugăm butoanele
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.start_button)
        button_layout2.addWidget(self.stop_button)

        # Adăugăm totul la layout-ul principal
        layout.addWidget(self.order_group_box)
        layout.addWidget(self.equations_group_box)
        layout.addWidget(self.hyperparameters_group_box)
        layout.addWidget(self.domain_group_box)
        layout.addWidget(self.file_path_button)
        layout.addLayout(button_layout2)

        # Setăm layout-ul principal
        self.setLayout(layout)

    def update_equations(self):
        """
        Actualizează numărul de ecuații pe baza ordinului sistemului.
        """
        order = self.system_order_input.value()

        # Elimina toate câmpurile de ecuație existente
        for eq_input in self.equation_inputs:
            eq_input.deleteLater()
        self.equation_inputs.clear()

        # Adaugă câte câmpuri de ecuație sunt necesare
        for _ in range(order):
            self.add_equation_input()

    def add_equation_input(self):
        """
        Adaugă un nou câmp de text pentru o ecuație.
        """
        equation_input = QLineEdit(self)
        equation_input.setPlaceholderText("Enter equation (e.g., dx1/dt)")
        equation_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # Mai flexibil
        self.equations_layout.addWidget(equation_input)
        self.equation_inputs.append(equation_input)

    def validate_domain(self):
        """
        Validăm că x_min nu este mai mare decât x_max.
        """
        if self.x_min_input.value() > self.x_max_input.value():
            self.x_min_input.setValue(self.x_max_input.value())

    def select_save_path(self):
        """
        Deschide un dialog pentru selectarea căii de salvare.
        """
        save_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if save_path:
            print(f"Selected path: {save_path}")

    def start_training(self):
        """
        Începe procesul de antrenare. Poți adăuga logica reală de antrenare.
        """
        print("Starting training...")
        self.is_training = True

        # Dezactivăm butonul Start și activăm Stop
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Blocăm câmpurile de ecuație pentru a preveni editarea lor
        self.set_equations_enabled(False)

        # Blocăm parametrii
        self.set_parameters_enabled(False)

    def stop_training(self):
        """
        Oprește procesul de antrenare.
        """
        print("Stopping training...")
        self.is_training = False

        # Reactivăm butonul Start și dezactivăm Stop
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Permitem modificarea parametrilor
        self.set_parameters_enabled(True)

        # Permitem editarea câmpurilor de ecuație
        self.set_equations_enabled(True)

    def set_parameters_enabled(self, enabled):
        """
        Permite sau blochează modificarea parametrilor în timpul antrenării.
        """
        self.learning_rate_input.setEnabled(enabled)
        self.input_size_input.setEnabled(enabled)
        self.hidden_layer_size_input.setEnabled(enabled)
        self.alpha_input.setEnabled(enabled)
        self.step_input.setEnabled(enabled)
        self.x_min_input.setEnabled(enabled)
        self.x_max_input.setEnabled(enabled)
        self.system_order_input.setEnabled(enabled)

    def set_equations_enabled(self, enabled):
        """
        Permite sau blochează modificarea câmpurilor de ecuație în timpul antrenării.
        """
        for eq_input in self.equation_inputs:
            eq_input.setEnabled(enabled)
