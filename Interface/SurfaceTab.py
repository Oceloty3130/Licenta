# Interface/SurfaceTab.py

import os, re, csv
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QPushButton, QComboBox, QRadioButton, QSizePolicy, QLabel, QFileDialog
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class SurfaceTab(QWidget):
    def __init__(self, minimum_graph_size, maximum_graph_size, parent=None, param_tab=None):
        super().__init__(parent)
        self.param_tab = param_tab
        self.setMinimumSize(minimum_graph_size)
        self.setMaximumSize(maximum_graph_size)

        # model name -> list of epoch CSV paths
        self.model_epochs = {}

        self._setup_ui()
        # Signal connections
        self.btn_folder.clicked.connect(self.load_model_folder)
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        self.combo_epoch.currentIndexChanged.connect(self.update_graph)
        self.rb_2d.toggled.connect(self.update_graph)
        self.rb_3d.toggled.connect(self.update_graph)
        self.rb_vx.toggled.connect(self.update_graph)
        self.rb_dvx.toggled.connect(self.update_graph)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Matplotlib canvas
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left = QWidget()
        left.setLayout(QVBoxLayout())
        left.layout().setContentsMargins(0, 0, 0, 0)
        left.layout().addWidget(self.canvas)
        main_layout.addWidget(left, 3)

        # Controls
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setAlignment(Qt.AlignTop)

        # Load folder and model
        grp1 = QGroupBox("Select Model Folder and Model")
        l1 = QVBoxLayout()
        self.btn_folder = QPushButton("Select Folder…")
        self.combo_model = QComboBox()
        l1.addWidget(self.btn_folder)
        l1.addWidget(self.combo_model)
        grp1.setLayout(l1)
        ctrl_layout.addWidget(grp1)

        # Epoch selection
        grp2 = QGroupBox("Select Epoch")
        l2 = QVBoxLayout()
        self.combo_epoch = QComboBox()
        l2.addWidget(self.combo_epoch)
        grp2.setLayout(l2)
        ctrl_layout.addWidget(grp2)

        # Display value
        grp3 = QGroupBox("Display Value")
        l3 = QVBoxLayout()
        self.rb_vx = QRadioButton("V(x)")
        self.rb_dvx = QRadioButton("DV(x)")
        self.rb_vx.setChecked(True)
        l3.addWidget(self.rb_vx)
        l3.addWidget(self.rb_dvx)
        grp3.setLayout(l3)
        ctrl_layout.addWidget(grp3)

        # Display mode
        grp4 = QGroupBox("Display Mode")
        l4 = QVBoxLayout()
        self.rb_2d = QRadioButton("2D")
        self.rb_3d = QRadioButton("3D")
        self.rb_2d.setChecked(True)
        l4.addWidget(self.rb_2d)
        l4.addWidget(self.rb_3d)
        grp4.setLayout(l4)
        ctrl_layout.addWidget(grp4)

        # Loss display
        self.loss_label = QLabel("Loss: N/A")
        self.loss_label.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(self.loss_label)

        # Update button
        self.btn_update = QPushButton("Update Graph")
        self.btn_update.clicked.connect(self.update_graph)
        ctrl_layout.addWidget(self.btn_update)

        ctrl_layout.addStretch()
        ctrl = QWidget()
        ctrl.setLayout(ctrl_layout)
        ctrl.setMinimumWidth(250)
        main_layout.addWidget(ctrl, 1)

        self.setLayout(main_layout)

    @Slot()
    def load_model_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select parent folder with models", "")
        if not folder:
            return
        self.model_epochs.clear()
        self.combo_model.clear()
        self.combo_epoch.clear()
        self.loss_label.setText("Loss: N/A")
        for entry in sorted(os.listdir(folder)):
            sub = os.path.join(folder, entry)
            if os.path.isdir(sub):
                epochs = [os.path.join(sub, f) for f in os.listdir(sub)
                          if f.startswith("training_data_epoch_") and f.endswith(".csv")]
                if epochs:
                    epochs.sort(key=lambda p: int(re.findall(r"epoch_(\d+)\.csv$", p)[0]))
                    self.model_epochs[entry] = epochs
                    self.combo_model.addItem(entry)
        if self.combo_model.count() > 0:
            self.combo_model.setCurrentIndex(0)

    @Slot(int)
    def on_model_changed(self, idx):
        self.combo_epoch.clear()
        self.loss_label.setText("Loss: N/A")
        model = self.combo_model.currentText()
        epochs = self.model_epochs.get(model, [])
        for path in epochs:
            num = re.findall(r"epoch_(\d+)\.csv$", path)[0]
            self.combo_epoch.addItem(f"Epoch {num}", path)
        if self.combo_epoch.count() > 0:
            self.combo_epoch.setCurrentIndex(0)
            self.update_graph()

    @Slot()
    def update_graph(self):
        idx = self.combo_epoch.currentIndex()
        if idx < 0:
            return
        epoch_path = self.combo_epoch.itemData(idx)
        if not epoch_path or not os.path.isfile(epoch_path):
            return

        # Load numeric CSV data for graph
        numeric = []
        with open(epoch_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not re.match(r'^[-+.]?\d', line):
                    continue
                numeric.append(line + '\n')
        data = np.loadtxt(numeric, delimiter=',')

        order = data.shape[1] - 2
        coords = data[:, :order]
        V_full = data[:, order]
        DV_full = data[:, order+1]

        # Update loss label by reading metrics CSV
        metrics_path = os.path.join(os.path.dirname(epoch_path), 'training_metrics.csv')
        epoch_num = int(re.findall(r"epoch_(\d+)\.csv$", epoch_path)[0])
        loss_val = None
        try:
            with open(metrics_path, 'r') as mf:
                reader = csv.reader(mf)
                next(reader, None)
                for e, l in reader:
                    if int(e) == epoch_num:
                        loss_val = float(l)
                        break
        except Exception:
            loss_val = None
        self.loss_label.setText(f"Loss: {loss_val:.6f}" if loss_val is not None else "Loss: N/A")

        # Prepare figure
        self.fig.clf()

        # Plotting logic same as before
        if order == 2:
            M = int(np.sqrt(coords.shape[0]))
            X_vals = coords[:,0].reshape(M, M)
            Y_vals = coords[:,1].reshape(M, M)
            val_arr = V_full.reshape(M, M) if self.rb_vx.isChecked() else DV_full.reshape(M, M)
            if self.rb_3d.isChecked():
                ax = self.fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X_vals, Y_vals, val_arr, cmap='viridis', edgecolor='none')
                self.fig.colorbar(surf, ax=ax, shrink=0.5)
            else:
                ax = self.fig.add_subplot(111)
                im = ax.imshow(val_arr,
                               extent=[X_vals.min(), X_vals.max(), Y_vals.min(), Y_vals.max()],
                               origin='lower', cmap='viridis', aspect='auto')
                self.fig.colorbar(im, ax=ax)
            ax.set_xlabel('x1'); ax.set_ylabel('x2')

        elif order == 1:
            ax = self.fig.add_subplot(111)
            x_vals = coords[:,0]
            if self.rb_vx.isChecked():
                ax.plot(x_vals, V_full, label='V(x)')
            else:
                ax.plot(x_vals, DV_full, label='DV(x)')
            ax.set_xlabel('x')
            ax.set_ylabel(self.rb_vx.text() if self.rb_vx.isChecked() else self.rb_dvx.text())
            ax.legend()

        else:
            mask = np.ones(coords.shape[0], dtype=bool)
            for d in range(2, order):
                mask &= (coords[:,d] == coords[:,d].min())
            subset = coords[mask]
            val_sub = V_full[mask] if self.rb_vx.isChecked() else DV_full[mask]
            x1_vals = np.unique(subset[:,0])
            x2_vals = np.unique(subset[:,1])
            M1, M2 = len(x1_vals), len(x2_vals)
            grid_val = np.full((M1, M2), np.nan)
            for (x1, x2, *_), v in zip(subset, val_sub):
                i = np.where(x1_vals==x1)[0][0]
                j = np.where(x2_vals==x2)[0][0]
                grid_val[i,j] = v
            ax = self.fig.add_subplot(111)
            im = ax.imshow(grid_val,
                           extent=[x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()],
                           origin='lower', cmap='viridis', aspect='auto')
            self.fig.colorbar(im, ax=ax)
            ax.set_xlabel('x1'); ax.set_ylabel('x2')

        # Title and redraw
        model_name = self.combo_model.currentText()
        epoch_label = self.combo_epoch.currentText()
        ax.set_title(f"{model_name} — {epoch_label}")
        self.canvas.draw()
