# Interface/main.py

import sys
from PySide6.QtWidgets import QApplication, QTabWidget, QMessageBox
from PySide6.QtCore import QSize, QThread, Qt
from Interface.ParametersTab import ParametersTab
from Interface.SurfaceTab    import SurfaceTab
from Interface.TrainingWorker import TrainingWorker

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()

        # Calculează dimensiunile minime/maxime pentru tab-uri
        screen_size = self.screen().size()
        minimum_graph_size = QSize(
            int(screen_size.width()  / 2),
            int(screen_size.height() / 1.75)
        )
        maximum_graph_size = screen_size

        # Creează tab-urile
        self.param_tab = ParametersTab(
            minimum_graph_size,
            maximum_graph_size,
            parent=self
        )
        self.surface_tab = SurfaceTab(
            minimum_graph_size,
            maximum_graph_size,
            parent=self,
            param_tab=self.param_tab
        )

        # Adaugă-le la QTabWidget
        self.addTab(self.param_tab,   "Parametri")
        self.addTab(self.surface_tab, "Surface")

        # Setează titlul și dimensiunea ferestrei
        self.setWindowTitle("Lyapunov Model")
        self.resize(1000, 800)

        # Conectează semnalul de start training
        self.param_tab.training_requested.connect(self.start_training)

    def start_training(self, payload: dict):
        """
        Slot apelat când ParametersTab emite training_requested.
        Pornește TrainingWorker pe un QThread.
        """
        params = payload.get('params', {})
        model_dir = payload.get('model_dir')

        if not model_dir:
            QMessageBox.critical(
                self,
                "Start Error",
                "Model directory not specified."
            )
            return

        # Inițializează worker și thread
        self.worker = TrainingWorker(params, model_dir)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # Conectează semnalele
        self.thread.started.connect(self.worker.run)
        self.worker.epoch_done.connect(self.surface_tab.update_graph)
        self.worker.finished.connect(self.on_training_finished)
        self.param_tab.stop_button.clicked.connect(self.worker.stop)

        # Curățare după terminare
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.param_tab.stop_training)

        # Pornește antrenarea în thread separat
        self.thread.start()

    def on_training_finished(self):
        """
        Slot apelat când Worker emite finished().
        Anunță utilizatorul că antrenarea s-a terminat.
        """
        QMessageBox.information(
            self,
            "Training Completed",
            "Training completed successfully and model was saved."
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
