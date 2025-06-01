from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QProgressBar, QPlainTextEdit
from defect_classification import run_training

class TrainingWorker(QObject):
    finished = pyqtSignal(object, str, list, list, list, tuple)
    update_text = pyqtSignal(str)
    update_status = pyqtSignal(str)

    def __init__(self, progress_bar, output_box, text_signal, status_signal):
        super().__init__()
        self.progress_bar = progress_bar
        self.output_box = output_box
        self.text_signal = text_signal
        self.status_signal = status_signal

    def run(self):
        history, base_path, data_set, sample_images, augmented_images, cm_data = run_training(
            progress_bar=self.progress_bar,
            text_output=self.output_box,
            text_signal=self.text_signal,
            status_signal=self.status_signal
        )
        self.finished.emit(history, base_path, data_set, sample_images, augmented_images, cm_data)