"""애플리케이션 설정 저장/로드 모듈 (QSettings 기반, 프로그램 종료 후에도 유지)"""
from PySide6.QtCore import QSettings

ORG_NAME = "KoreaHRD"
APP_NAME = "IDPhotoAutoEditor"


class AppSettings:
    def __init__(self):
        self._settings = QSettings(ORG_NAME, APP_NAME)

    def get_last_save_folder(self) -> str:
        return self._settings.value("last_save_folder", "", type=str)

    def set_last_save_folder(self, folder: str) -> None:
        self._settings.setValue("last_save_folder", folder)
        self._settings.sync()
