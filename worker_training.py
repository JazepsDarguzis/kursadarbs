from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QProgressBar, QPlainTextEdit
from defect_classification import run_training  # твоя функция

class TrainingWorker(QObject):
    finished = pyqtSignal(object, str, list)  # history, base_path, data_set

    def run(self):
        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        history, base_path, data_set = run_training(progress_bar=self.progress_bar, text_output=self.output_box)
        self.finished.emit(history, base_path, data_set)
