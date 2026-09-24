# -*- coding: utf-8 -*-
"""
motor_facial.py
---------------
Envoltorio sobre InsightFace: detecta rostros y genera embeddings.

NO se entrena ningun modelo. Se usa el paquete preentrenado 'buffalo_l' de
InsightFace, que internamente ejecuta dos redes con ONNX Runtime:

  1. Un detector (SCRFD) que encuentra los rostros de la imagen.
  2. Un extractor (ArcFace) que convierte cada rostro en un vector de 512
     numeros llamado *embedding*.

La idea clave del reconocimiento facial moderno: dos fotos de la MISMA persona
producen embeddings que apuntan casi en la misma direccion, y dos personas
distintas producen embeddings que apuntan en direcciones diferentes. Por eso
comparar caras se reduce a medir el angulo entre dos vectores (similitud
coseno).
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class ErrorModeloFacial(Exception):
    """Se lanza cuando el modelo preentrenado no se puede cargar."""


@dataclass
class RostroDetectado:
    """Resultado de analizar un rostro dentro de un frame."""

    caja: tuple           # (x1, y1, x2, y2) en pixeles, ya recortada al frame
    embedding: np.ndarray  # vector de 512 numeros, normalizado (norma = 1)
    confianza_deteccion: float
    nitidez: float         # varianza del Laplaciano del recorte
    calidad: float         # puntuacion combinada entre 0 y 1

    @property
    def ancho(self):
        return self.caja[2] - self.caja[0]

    @property
    def alto(self):
        return self.caja[3] - self.caja[1]


# ---------------------------------------------------------------------------
# FUNCIONES DE APOYO (independientes del modelo)
# ---------------------------------------------------------------------------


def normalizar(vector):
    """
    Devuelve el vector con norma 1.

    Trabajar siempre con vectores normalizados permite calcular la similitud
    coseno como un simple producto punto, que es mucho mas rapido.
    """
    vector = np.asarray(vector, dtype=np.float32)
    norma = np.linalg.norm(vector)
    if norma == 0:
        return vector
    return vector / norma


def similitud_coseno(vector_a, vector_b):
    """
    Similitud coseno entre dos embeddings: va de -1 a 1.

    1.0  = identicos
    ~0.5 = probablemente la misma persona (depende del umbral calibrado)
    ~0.0 = personas distintas
    """
    return float(np.dot(normalizar(vector_a), normalizar(vector_b)))


def medir_nitidez(imagen, ancho_referencia=300):
    """
    Mide lo nitida que es una imagen con la varianza del Laplaciano.

    El Laplaciano resalta los bordes: una foto enfocada tiene bordes marcados
    (varianza alta) y una foto movida o borrosa los tiene suaves (varianza
    baja). Sirve para descartar capturas malas antes de guardarlas.

    IMPORTANTE - la metrica NO es invariante a la escala: en una foto de
    celular el mismo rostro ocupa 3-5 veces mas pixeles que en la webcam, y
    sus bordes quedan 'suaves por pixel' aunque este perfectamente enfocado
    (la varianza se derrumba y daba falsos 'borrosa'). Por eso la medicion
    se normaliza: si el recorte es mas ancho que el ancho de referencia
    (300 px, el tamano tipico de los recortes de la camara con los que se
    calibro MINIMA_NITIDEZ), se reduce antes de medir.
    """
    if imagen is None or imagen.size == 0:
        return 0.0
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if gris.shape[1] > ancho_referencia:
        escala = ancho_referencia / gris.shape[1]
        nuevo_alto = max(1, int(gris.shape[0] * escala))
        gris = cv2.resize(gris, (ancho_referencia, nuevo_alto),
                          interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def _calcular_calidad(confianza_deteccion, nitidez):
    """
    Combina confianza del detector y nitidez en una puntuacion 0..1.

    Es una heuristica sencilla y transparente: la mitad del peso a lo seguro
    que esta el detector de que hay una cara, y la otra mitad a lo enfocada
    que esta la imagen (saturando la nitidez en 200, valor a partir del cual
    ya se considera claramente nitida).
    """
    nitidez_normalizada = min(1.0, nitidez / 200.0)
    return round(0.5 * float(confianza_deteccion) + 0.5 * nitidez_normalizada, 4)


# ---------------------------------------------------------------------------
# MOTOR FACIAL
# ---------------------------------------------------------------------------


class MotorFacial:
    """
    Carga el modelo preentrenado una sola vez y ofrece deteccion + embeddings.

    La carga es lenta (varios segundos la primera vez, porque ademas puede
    descargar el modelo), asi que se hace una unica instancia y se comparte.
    """

    def __init__(self, cargar_ahora=True):
        self._app = None
        if cargar_ahora:
            self.cargar()

    def cargar(self):
        """
        Prepara InsightFace. Lanza ErrorModeloFacial con un mensaje entendible
        si algo falla (falta la libreria, no hay modelo y no hay Internet...).
        """
        if self._app is not None:
            return

        config.crear_directorios()

        try:
            # La importacion se hace aqui dentro y no arriba para que el resto
            # del proyecto (base de datos, pruebas) se pueda usar aunque
            # InsightFace todavia no este instalado.
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise ErrorModeloFacial(
                "No se pudo importar InsightFace. Instala las dependencias "
                "con: pip install -r requirements.txt"
            ) from error

        try:
            logger.info("Cargando modelo facial '%s'...", config.NOMBRE_MODELO)
            app = FaceAnalysis(
                name=config.NOMBRE_MODELO,
                root=str(config.DIRECTORIO_MODELOS),
                providers=config.PROVEEDORES_ONNX,
            )
            # ctx_id = -1 significa CPU. Con GPU seria 0.
            app.prepare(ctx_id=-1, det_size=config.TAMANO_DETECCION)
        except Exception as error:  # noqa: BLE001 - se re-lanza traducido
            raise ErrorModeloFacial(
                "No se pudo cargar el modelo facial preentrenado.\n"
                "Causa: {}\n\n"
                "La primera ejecucion necesita Internet UNA sola vez para "
                "descargar el modelo en la carpeta 'modelos/'. Despues el "
                "programa funciona totalmente sin conexion.".format(error)
            ) from error

        self._app = app
        logger.info("Modelo facial cargado correctamente")

    @property
    def esta_cargado(self):
        return self._app is not None

    # -- Deteccion ----------------------------------------------------------

    def detectar(self, frame):
        """
        Analiza un frame BGR de OpenCV y devuelve la lista de RostroDetectado.

        Devuelve lista vacia si no hay ninguna cara.
        """
        if self._app is None:
            raise ErrorModeloFacial("El modelo facial no esta cargado")
        if frame is None or frame.size == 0:
            return []

        rostros = self._analizar_frame(frame)

        # 2do proyecto del temario (mejoramiento de imagen): si el frame esta
        # muy degradado (oscuro, ruidoso, pequenio) y no aparece ningun rostro,
        # se reintenta UNA sola vez con la imagen mejorada por el pipeline
        # clasico de mejorar_imagen.py. Nunca altera un frame donde ya se
        # detecto algo.
        if not rostros and config.REINTENTAR_DETECCION_MEJORANDO:
            import mejorar_imagen
            rostros = self._analizar_frame(mejorar_imagen.mejorar(frame, modo="deteccion"))

        return rostros

    def _analizar_frame(self, frame):
        """Pasa UN frame por el detector y construye los RostroDetectado."""
        alto_frame, ancho_frame = frame.shape[:2]
        rostros = []

        for cara in self._app.get(frame):
            # bbox llega como float; se pasa a enteros y se recorta a los
            # limites del frame para que los recortes nunca salgan vacios.
            x1, y1, x2, y2 = [int(valor) for valor in cara.bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(ancho_frame, x2)
            y2 = min(alto_frame, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            recorte = frame[y1:y2, x1:x2]
            nitidez = medir_nitidez(recorte)
            confianza = float(getattr(cara, "det_score", 0.0))

            # normed_embedding ya viene con norma 1 desde InsightFace; se
            # vuelve a normalizar por seguridad para no depender de detalles
            # internos de la version instalada.
            embedding = normalizar(cara.normed_embedding)

            rostros.append(
                RostroDetectado(
                    caja=(x1, y1, x2, y2),
                    embedding=embedding,
                    confianza_deteccion=confianza,
                    nitidez=nitidez,
                    calidad=_calcular_calidad(confianza, nitidez),
                )
            )

        return rostros

    # -- Validacion de calidad para el registro ------------------------------

    def obtener_rostro_para_captura(self, frame):
        """
        Detecta y valida en un solo paso. Atajo de validar_rostro_unico().
        """
        return self.validar_rostro_unico(self.detectar(frame))

    @staticmethod
    def validar_rostro_unico(rostros):
        """
        Valida una lista ya detectada pensando en GUARDAR una muestra.

        Devuelve (rostro, motivo). Si 'rostro' es None, 'motivo' explica en
        espanol por que no se acepta la captura, y ese texto se muestra al
        usuario en la interfaz.

        Las reglas son deliberadamente estrictas: una muestra mala contamina
        el perfil de la persona para siempre.

        Recibe la lista ya detectada (en lugar del frame) para que la interfaz
        pueda detectar UNA vez por frame y reutilizar el resultado tanto para
        dibujar como para validar.
        """
        if not rostros:
            return None, "No se detecta ningun rostro"
        if len(rostros) > 1:
            return None, "Se detecta mas de un rostro: debe haber solo uno"

        rostro = rostros[0]

        if rostro.confianza_deteccion < config.MINIMA_CONFIANZA_DETECCION:
            return None, "Deteccion poco fiable, mejora la iluminacion"
        if (rostro.ancho < config.MINIMO_ANCHO_ROSTRO
                or rostro.alto < config.MINIMO_ALTO_ROSTRO):
            return None, "Rostro demasiado pequeno, acercate a la camara"
        if rostro.nitidez < config.MINIMA_NITIDEZ:
            return None, "Imagen borrosa, quedate quieto un momento"

        return rostro, None
