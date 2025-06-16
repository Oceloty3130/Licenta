import sys

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QTabWidget

from Interface.ParametersTab import ParametersTab
from Interface.SurfaceTab import SurfaceTab


class MainWindow(QTabWidget):
    def __init__(self, p=None):
        super().__init__(p)
        screen_size = self.screen().size()
        minimum_graph_size = QSize(screen_size.width() / 2, screen_size.height() / 1.75)

        self._param_tab = ParametersTab(minimum_graph_size, screen_size)
        self._surface_tab = SurfaceTab(minimum_graph_size, screen_size, self._param_tab)

        # Adăugăm tab-urile
        self.addTab(self._param_tab, "Parametri")
        self.addTab(self._surface_tab, "Surface")

        # Ajustează dimensiunile și layout-ul
        self.adjustSize()  # Ajustează automat dimensiunile la conținut



if __name__ == "__main__":
    app = QApplication(sys.argv)

    tabWidget = MainWindow()
    tabWidget.setWindowTitle("Lyapunov Model")

    tabWidget.resize(1000, 800)
    tabWidget.show()
    exit_code = app.exec()
    del tabWidget
    sys.exit(exit_code)
