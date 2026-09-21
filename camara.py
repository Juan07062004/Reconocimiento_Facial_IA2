# -*- coding: utf-8 -*-
"""
camara.py
---------
Acceso a la camara con OpenCV, aislado en un solo modulo.

Todo el manejo de errores de camara (no existe, esta ocupada por otro
programa, se desconecta a mitad) vive aqui, para que el resto del proyecto
solo tenga que preocuparse de recibir frames.
"""

import logging
import sys

import cv2

import config

logger = logging.getLogger(__name__)


class ErrorCamara(Exception):
    """Se lanza cuando la camara no se puede abrir o deja de responder."""


class Camara:
    """
    Envoltorio sencillo sobre cv2.VideoCapture.

    Se puede usar como gestor de contexto para garantizar que la camara
    SIEMPRE se libera, incluso si ocurre un error:

        with Camara() as camara:
            frame = camara.leer()
    """

    def __init__(self, indice=None, ancho=None, alto=None):
        self.indice = config.INDICE_CAMARA if indice is None else indice
        self.ancho = config.ANCHO_CAMARA if ancho is None else ancho
        self.alto = config.ALTO_CAMARA if alto is None else alto
        self._captura = None

    # -- Ciclo de vida -------------------------------------------------------

    def abrir(self):
        """Abre la camara. Lanza ErrorCamara si no hay ninguna disponible."""
        if self.esta_abierta:
            return

        # En Windows, DirectShow (CAP_DSHOW) evita el retardo de varios
        # segundos que introduce el backend por defecto al abrir la camara.
        if sys.platform.startswith("win"):
            captura = cv2.VideoCapture(self.indice, cv2.CAP_DSHOW)
        else:
            captura = cv2.VideoCapture(self.indice)

        if not captura.isOpened():
            captura.release()
            raise ErrorCamara(
                "No se pudo abrir la camara numero {}.\n\n"
                "Revisa que:\n"
                "  - El computador tenga camara conectada.\n"
                "  - Ningun otro programa la este usando "
                "(Zoom, Teams, Meet, el navegador...).\n"
                "  - Windows tenga permitido el acceso a la camara en "
                "Privacidad > Camara.\n"
                "  - Si tienes varias camaras, prueba a cambiar "
                "INDICE_CAMARA en config.py.".format(self.indice)
            )

        captura.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)

        self._captura = captura
        logger.info("Camara %s abierta", self.indice)

    def liberar(self):
        """Libera la camara. Es seguro llamarla varias veces."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None
            logger.info("Camara %s liberada", self.indice)

    @property
    def esta_abierta(self):
        return self._captura is not None and self._captura.isOpened()

    # -- Lectura -------------------------------------------------------------

    def leer(self, espejo=True):
        """
        Devuelve el siguiente frame (BGR) o None si la lectura fallo.

        'espejo' voltea la imagen horizontalmente para que el usuario se vea
        como en un espejo, que es lo natural al mirarse en pantalla. El
        volteo no afecta al reconocimiento.
        """
        if not self.esta_abierta:
            return None

        exito, frame = self._captura.read()
        if not exito or frame is None:
            return None

        if espejo:
            frame = cv2.flip(frame, 1)
        return frame

    # -- Gestor de contexto --------------------------------------------------

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, tipo_error, valor_error, traza):
        self.liberar()
        return False  # no se traga las excepciones

    def __del__(self):
        # Ultima red de seguridad por si alguien olvida liberar la camara.
        try:
            self.liberar()
        except Exception:  # noqa: BLE001 - nunca fallar dentro de __del__
            pass


def listar_camaras_disponibles(maximo=5):
    """
    Prueba los indices 0..maximo-1 y devuelve los que responden.

    Util para diagnosticar cuando el portatil tiene camara integrada y ademas
    una camara USB conectada.
    """
    disponibles = []
    for indice in range(maximo):
        if sys.platform.startswith("win"):
            captura = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        else:
            captura = cv2.VideoCapture(indice)
        if captura.isOpened():
            exito, _ = captura.read()
            if exito:
                disponibles.append(indice)
        captura.release()
    return disponibles
