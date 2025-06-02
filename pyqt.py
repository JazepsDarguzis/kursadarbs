import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QPlainTextEdit, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import cv2
from worker_training import TrainingWorker
import numpy as np
import seaborn as sns
from defect_classification import run_training
from matplotlib.figure import Figure


class HomePage(QWidget):
    update_text_signal = pyqtSignal(str)
    stats_ready_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.process = None
        self.update_text_signal.connect(self.update_text)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("🏠 Добро пожаловать в систему анализа дефектов"))
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Запустить анализ дефектов")
        self.run_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.run_button)
        layout.addLayout(button_layout)
        self.output_box = QPlainTextEdit()
        self.output_box.setReadOnly(True)
        layout.addWidget(self.output_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def update_text(self, text):
        self.output_box.appendPlainText(text)
        self.output_box.repaint()

    def start_training(self):
        self.run_button.setEnabled(False)
        self.run_button.setText("Обучение...")
        self.output_box.clear()
        self.progress_bar.setValue(0)
        self.thread = QThread()
        self.worker = TrainingWorker(self.progress_bar, self.output_box, self.update_text_signal)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_training_finished(self, history, base_path, data_set, sample_images, augmented_images, cm_data):
        self.run_button.setEnabled(True)
        self.run_button.setText("Запустить анализ дефектов")
        self.stats_ready_signal.emit(history)
        self.parent().findChild(MonitoringPage1, "MonitoringPage1").show_sample_images(sample_images)
        self.parent().findChild(MonitoringPage2, "MonitoringPage2").show_augmented_images(augmented_images)
        self.parent().findChild(DecisionMakingPage, "DecisionMakingPage").show_confusion_matrix(cm_data)

class StatisticsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatisticsPage")
        self.layout = QVBoxLayout()
        self.history = None
        self.is_shown = False  # Флаг, чтобы графики отображались только один раз
        self.setLayout(self.layout)

    def on_tab_changed(self, index):
        if index == 1 and self.history is not None and not self.is_shown:
            self.show_training_statistics(self.history)
            self.is_shown = True

    def show_training_statistics(self, history):
        self.history = history
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history.history['accuracy'], label='Точность (обучение)')
        ax[0].plot(history.history['val_accuracy'], label='Точность (валидация)')
        ax[0].legend()
        ax[0].set_title('Точность')
        ax[1].plot(history.history['loss'], label='Потери (обучение)')
        ax[1].plot(history.history['val_loss'], label='Потери (валидация)')
        ax[1].legend()
        ax[1].set_title('Потери')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        self.layout.addWidget(canvas)

class MonitoringPage1(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MonitoringPage1")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def show_sample_images(self, sample_images):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        num_images = min(5, len(sample_images))
        fig, ax = plt.subplots(1, num_images, figsize=(3 * num_images, 3))
        if num_images == 1:
            ax = [ax]
        for i, (img_path, class_name) in enumerate(sample_images[:num_images]):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
            ax[i].set_title(f"Класс: {class_name}")
            ax[i].axis('off')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        self.layout.addWidget(canvas)

class MonitoringPage2(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MonitoringPage2")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def show_augmented_images(self, augmented_images):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        num_images = min(5, len(augmented_images))
        fig, ax = plt.subplots(1, num_images, figsize=(3 * num_images, 3))
        if num_images == 1:
            ax = [ax]
        for i, (img, class_name) in enumerate(augmented_images[:num_images]):
            img = img.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax[i].imshow(img)
            ax[i].set_title(f"Класс: {class_name}")
            ax[i].axis('off')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        self.layout.addWidget(canvas)

class DecisionMakingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DecisionMakingPage")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def show_confusion_matrix(self, cm_data):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        cm, class_names = cm_data
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Матрица ошибок')
        ax.set_xlabel('Предсказано')
        ax.set_ylabel('Истина')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        self.layout.addWidget(canvas)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система анализа дефектов")
        self.setGeometry(100, 100, 800, 600)
        self.tabs = QTabWidget()
        self.home_page = HomePage()
        self.stats_page = StatisticsPage()
        self.monitoring_page1 = MonitoringPage1()
        self.monitoring_page2 = MonitoringPage2()
        self.decision_page = DecisionMakingPage()

        self.tabs.addTab(self.home_page, "Главная")
        self.tabs.addTab(self.stats_page, "Статистика")
        self.tabs.addTab(self.monitoring_page1, "Мониторинг 1")
        self.tabs.addTab(self.monitoring_page2, "Мониторинг 2")
        self.tabs.addTab(self.decision_page, "Принятие решений")

        # Подключаем сигнал для обновления StatisticsPage
        self.home_page.stats_ready_signal.connect(self.stats_page.show_training_statistics)
        self.tabs.currentChanged.connect(self.stats_page.on_tab_changed)

        self.setCentralWidget(self.tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())