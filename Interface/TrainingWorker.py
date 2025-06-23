# Interface/TrainingWorker.py

import os
import json
import csv
import torch
import torch.optim as optim
import numpy as np

from PySide6.QtCore import QObject, Signal, Slot, QCoreApplication
from Neural_Network.Lyapunov_Model import (
    LyapunovNet,
    custom_lyap_loss,
    state_fcn,
    save_model_with_encrypted_hash
)


class TrainingWorker(QObject):
    """
    Worker pentru antrenarea LyapunovNet pe un QThread.
    Emite epoch_done(epoch, loss, V_vals, DV_vals) și finished().
    """

    # slot pentru oprirea antrenării
    @Slot()
    def stop(self):
        """Stop the training loop after current epoch."""
        self._is_running = False

    epoch_done = Signal(int, float, np.ndarray, np.ndarray)
    finished = Signal()
    """
    Worker pentru antrenarea LyapunovNet pe un QThread.
    Emite epoch_done(epoch, loss, V_vals, DV_vals) și finished().
    """
    epoch_done = Signal(int, float, np.ndarray, np.ndarray)
    finished = Signal()

    def __init__(self, params: dict, model_dir: str, parent=None):
        super().__init__(parent)
        self.params = params
        self._is_running = True
        # Corectăm dacă s-a dat folderul de securitate în loc de folderul modelului
        if os.path.basename(model_dir) == "Model_Parameters":
            model_dir = os.path.dirname(model_dir)
        # Setăm directorul efectiv al modelului
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        # Directoriu de securitate unic
        self.security_dir = os.path.join(self.model_dir, "Model_Parameters")
        os.makedirs(self.security_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Salvăm parametrii într-un JSON în acest folder
        with open(os.path.join(self.model_dir, "params.json"), "w") as fp:
            json.dump(params, fp, indent=2)

        # Pregătim CSV-urile
        self.metrics_csv = os.path.join(self.model_dir, "training_metrics.csv")
        with open(self.metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss"])

        self.data_csv = os.path.join(self.model_dir, "training_data.csv")
        if os.path.exists(self.data_csv):
            os.remove(self.data_csv)

    @Slot()
    def run(self):
        p = self.params
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exprs = p["expressions"]
        order = p["order"]
        lr = p["learning_rate"]
        hid_size = p["hidden_layer_size"]
        alpha = p["alpha"]
        x_min = p["x_min"]
        x_max = p["x_max"]
        step = p["step"]
        gamma = p.get("gamma", 0.01)
        max_epochs = p["max_epochs"]

        # CREĂM întotdeauna un model nou
        model = LyapunovNet(
            input_size=order,
            hidden_layer_size=hid_size,
            alpha=alpha
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()

        # Pregătim grila N-dimensională
        xs = torch.arange(x_min, x_max + 1e-9, step, device=device)
        grid_pts = torch.cartesian_prod(*[xs] * order)
        num_pts = grid_pts.size(0)
        M = xs.size(0)

        for epoch in range(1, max_epochs + 1):
            if not self._is_running:
                break
            QCoreApplication.processEvents()

            optimizer.zero_grad()
            loss = custom_lyap_loss(x_min, x_max, step, model, exprs, gamma)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} - Loss = {loss.item()}")

            loss_val = loss.item()
            # Salvăm metricile pe epocă
            with open(self.metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, loss_val])

            # Colectăm V și DV
            V_list = np.zeros(num_pts, dtype=float)
            DV_list = np.zeros(num_pts, dtype=float)
            for idx, pt in enumerate(grid_pts):
                x_pt = pt.clone().detach().view(1, -1).requires_grad_(True)
                V_list[idx] = model(x_pt).item()
                grads = torch.autograd.grad(model(x_pt), x_pt,
                                            retain_graph=True,
                                            create_graph=True)[0]
                f_val = state_fcn(exprs, x_pt)
                DV_list[idx] = float((grads * f_val).sum().item())

            # Scriem bloc CSV pentru epocă
            epoch_csv = os.path.join(
                self.model_dir, f"training_data_epoch_{epoch}.csv"
            )
            with open(epoch_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"Epoch {epoch}"] * (order + 2))
                writer.writerow([f"x{i + 1}" for i in range(order)] + ["Vx", "DVx"])
                coords = grid_pts.cpu().numpy()
                # asigurăm un array 2D chiar și pentru order=1
                if coords.ndim == 1:
                    coords = coords.reshape(-1, 1)
                for i in range(num_pts):
                    writer.writerow(list(coords[i]) + [V_list[i], DV_list[i]])

            # Emit pentru UI
            if order == 2:
                V_vals = V_list.reshape(M, M)
                DV_vals = DV_list.reshape(M, M)
            else:
                V_vals = V_list.reshape(-1, 1)
                DV_vals = DV_list.reshape(-1, 1)
            self.epoch_done.emit(epoch, loss_val, V_vals, DV_vals)

            # Salvare periodică la fiecare 10 epoci
            if epoch % 5 == 0:
                save_model_with_encrypted_hash(model, self.model_dir)
            if loss_val < 10e-4:
                break

        # Salvăm model final și securitate
        save_model_with_encrypted_hash(model, self.model_dir)
        self.finished.emit()
        sec_dir = os.path.join(self.model_dir, "Model_Parameters")
        os.makedirs(sec_dir, exist_ok=True)
        save_model_with_encrypted_hash(model, sec_dir)
        self.finished.emit()
