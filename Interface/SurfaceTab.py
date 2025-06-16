# Interface/SurfaceTab.py

import os, csv
import numpy as np
from PySide6.QtCore    import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QComboBox, QPushButton, QLabel, QSizePolicy, QFileDialog
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

import re
import numpy as np
import csv
from matplotlib.figure import Figure

class SurfaceTab(QWidget):
    def __init__(self, minimum_graph_size, maximum_graph_size, parent=None):
        super().__init__(parent)

        # hartă model_name -> cale_csv
        self.model_csv_map = {}

        # Layout principal
        main_layout = QHBoxLayout(self)

        # ─── Stânga: canvas matplotlib ───
        self.fig = Figure(figsize=(5,4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        # politici de redimensionare
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left = QWidget()
        left.setLayout(QVBoxLayout())
        left.layout().setContentsMargins(0,0,0,0)
        left.layout().addWidget(self.canvas)
        main_layout.addWidget(left, 3)

        # ─── Dreapta: controale ───
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setAlignment(Qt.AlignTop)

        # 1) folder părinte cu modele
        grp1 = QGroupBox("Select Models Folder")
        l1 = QVBoxLayout()
        self.btn_folder = QPushButton("Select Folder…")
        self.btn_folder.clicked.connect(self.select_folder)
        self.combo_model = QComboBox()
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        l1.addWidget(self.btn_folder)
        l1.addWidget(self.combo_model)
        grp1.setLayout(l1)
        ctrl_layout.addWidget(grp1)

        # 2) epocă
        grp2 = QGroupBox("Select Epoch")
        l2 = QVBoxLayout()
        self.combo_epoch = QComboBox()
        l2.addWidget(self.combo_epoch)
        grp2.setLayout(l2)
        ctrl_layout.addWidget(grp2)

        # 3) tip grafic
        grp3 = QGroupBox("Graph Type")
        l3 = QVBoxLayout()
        self.rb_vx = QRadioButton("V(x)")
        self.rb_dvx= QRadioButton("DV(x)")
        self.rb_vx.setChecked(True)
        l3.addWidget(self.rb_vx)
        l3.addWidget(self.rb_dvx)
        grp3.setLayout(l3)
        ctrl_layout.addWidget(grp3)

        # 4) 2D / 3D
        grp4 = QGroupBox("Display Mode")
        l4 = QVBoxLayout()
        self.rb_2d = QRadioButton("2D")
        self.rb_3d = QRadioButton("3D")
        self.rb_2d.setChecked(True)
        l4.addWidget(self.rb_2d)
        l4.addWidget(self.rb_3d)
        grp4.setLayout(l4)
        ctrl_layout.addWidget(grp4)

        # 5) buton actualizare
        self.btn_update = QPushButton("Update Graph")
        self.btn_update.clicked.connect(self.update_graph)
        ctrl_layout.addWidget(self.btn_update)

        ctrl_layout.addStretch()
        ctrl = QWidget()
        ctrl.setLayout(ctrl_layout)
        ctrl.setMinimumWidth(200)
        ctrl.setMaximumWidth(300)
        main_layout.addWidget(ctrl, 1)

        self.setLayout(main_layout)

    def select_folder(self):
        """Selectează folderul părinte și identifică CSV-urile din subfoldere."""
        parent = QFileDialog.getExistingDirectory(self, "Select parent folder with models")
        if not parent:
            return
        self.model_csv_map.clear()
        for entry in sorted(os.listdir(parent)):
            sub = os.path.join(parent, entry)
            if os.path.isdir(sub):
                # caută primul CSV
                for f in os.listdir(sub):
                    if f.lower().endswith(".csv"):
                        self.model_csv_map[entry] = os.path.join(sub, f)
                        break
        # populează combo_model
        self.combo_model.clear()
        self.combo_model.addItems(self.model_csv_map.keys())
        # încarcă header pentru primul
        if self.model_csv_map:
            first = next(iter(self.model_csv_map.values()))
            self.load_csv_header(first)

    def on_model_changed(self, idx):
        name = self.combo_model.currentText()
        path = self.model_csv_map.get(name)
        if path:
            self.load_csv_header(path)

    def load_csv_header(self, path):
        with open(path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
        self.current_header = header  # <— adăugat
        epochs = header[1:]
        self.combo_epoch.clear()
        self.combo_epoch.addItems(epochs)

    def update_graph(self):
        model = self.combo_model.currentText()
        path = self.model_csv_map.get(model)
        if not path:
            return

        # 1) Încarcă CSV-ul complet într-un array (skip header)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        X_full = data[:, 0]  # shape (n,)
        Z_full = data[:, 1:]  # shape (n, m)  — fiecare coloană e o epocă

        # 2) Vector numeric de epoci
        #    Extrage cifrele din string-urile header: ["epoch1","epoch2",...]
        E_labels = self.current_header[1:]
        E = []
        for lbl in E_labels:
            nums = [int(s) for s in re.findall(r"\d+", lbl)]
            E.append(nums[0] if nums else 0)
        E = np.array(E)  # shape (m,)

        # 3) meshgrid: notează că meshgrid produce shape (m, n)
        X_mesh, E_mesh = np.meshgrid(X_full, E, indexing='xy')
        Z_mesh = Z_full.T  # transpose => shape (m, n)

        # 4) Desenăm suprafața 3D
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X_mesh, E_mesh, Z_mesh,
            cmap="viridis", edgecolor="none"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Epoch")
        ax.set_zlabel("Value")
        ax.set_title(f"Surface for {model}")
        self.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        self.canvas.draw()