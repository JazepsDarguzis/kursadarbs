import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QPlainTextEdit, QProgressBar
)
from PyQt5.QtCore import QThread
from defect_classification import run_training  # Импортируем функцию из defect_classification.py
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2
import os
import random
from worker_training import TrainingWorker


# Главная вкладка
class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.process = None

        # Layout
        layout = QVBoxLayout()

        # Текст
        layout.addWidget(QLabel("🏠 Welcome to the Main Window"))

        # Горизонтальный блок: кнопка
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Запустить анализ дефектов")
        self.run_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.run_button)
        layout.addLayout(button_layout)

        # Output box
        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        layout.addWidget(self.output_box)

        # Прогрессбар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)


    def start_training(self):
        # блокируем кнопку
        self.run_button.setEnabled(False)
        self.run_button.setText("Обучение...")

        # создаём поток и воркера
        self.thread = QThread()
        self.worker = TrainingWorker()
        self.worker.moveToThread(self.thread)

        # подключаем сигналы
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # стартуем поток
        self.thread.start()

    def on_training_finished(self, history, base_path, data_set):
        # разблокируем кнопку
        self.run_button.setEnabled(True)
        self.run_button.setText("Начать обучение")

        # обновляем вкладки
        # self.statistics_page.show_training_statistics(history)
        # self.monitoring_page1.show_sample_images(base_path, data_set)


# Остальные вкладки (просто-заполнители)
class StatisticsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def show_training_statistics(self, history):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        ax[0].legend()
        ax[0].set_title('Accuracy')

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Val Loss')
        ax[1].legend()
        ax[1].set_title('Loss')

        canvas = FigureCanvas(fig)
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)
        self.setLayout(layout)


class MonitoringPage1(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def show_sample_images(self, base_path, data_set):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt
        import cv2
        import os
        import random

        fig, ax = plt.subplots(3, 3, figsize=(9, 8))
        for i in range(min(3, len(data_set))):
            folder = os.path.join(base_path, data_set[i])
            for j in range(3):
                file_name = random.choice(os.listdir(folder))
                img = cv2.imread(os.path.join(folder, file_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax[i, j].imshow(img)
                ax[i, j].set_title(f"Class: {data_set[i]}")
                ax[i, j].axis('off')

        canvas = FigureCanvas(fig)
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)
        self.setLayout(layout)

class MonitoringPage2(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("📡 Monitoring data 2..."))
        self.setLayout(layout)

class DecisionMakingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("🤖 Decision-making logic here..."))
        self.setLayout(layout)


# Главное окно
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Interface with Tabs")
        self.setGeometry(100, 100, 700, 500)

        self.tabs = QTabWidget()
        self.tabs.addTab(HomePage(), "Main")
        self.tabs.addTab(StatisticsPage(), "Statistics")
        self.tabs.addTab(MonitoringPage1(), "Monitoring 1")
        self.tabs.addTab(MonitoringPage2(), "Monitoring 2")
        self.tabs.addTab(DecisionMakingPage(), "Decision Making")

        self.setCentralWidget(self.tabs)


# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
