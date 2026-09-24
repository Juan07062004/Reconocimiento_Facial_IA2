# -*- coding: utf-8 -*-
"""
reconocer.py
------------
Logica de reconocimiento: dado el embedding de un rostro, decidir a quien
pertenece o declararlo DESCONOCIDO.

ESTRATEGIA DE PRECISION (esta es la parte importante del proyecto)
------------------------------------------------------------------
Una sola foto NO representa bien a una persona: cambia la luz, el angulo, las
gafas, la barba... Por eso cada persona guarda VARIOS embeddings.

Al comparar un rostro nuevo contra la base de datos:

  1. Se calcula la similitud coseno contra TODAS las muestras guardadas.
     Como todos los embeddings estan normalizados, esto es un unico producto
     matricial: rapido incluso con cientos de muestras.

  2. Las similitudes se agrupan POR PERSONA y se agregan en una sola
     puntuacion, segun config.ESTRATEGIA_AGREGACION:

       - "mejor": se queda con la muestra mas parecida (el maximo).
         Es sensible: una unica coincidencia afortunada basta para decidir,
         lo que sube los falsos positivos.

       - "top_k_media" (por defecto): promedia las TOP_K muestras mas
         parecidas de esa persona. Exige coherencia con varias muestras, no
         con una sola, y por eso es mas robusto frente a falsos positivos.
         Si la persona tiene menos de TOP_K muestras, se promedian todas.

  3. Gana la persona con mayor puntuacion, pero SOLO se acepta si esa
     puntuacion supera config.UMBRAL_RECONOCIMIENTO. Si no lo supera, el
     resultado es DESCONOCIDO: es preferible no identificar a nadie antes que
     identificar a la persona equivocada.

El umbral NO es un valor universal: depende del modelo, de la camara y de la
iluminacion. Calibralo con pruebas/calibrar_umbral.py.
"""

import logging
import unicodedata
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import base_datos
import config

logger = logging.getLogger(__name__)

# Colores BGR (OpenCV no usa RGB) para dibujar sobre el video.
COLOR_IDENTIFICADO = (0, 180, 0)      # verde
COLOR_ALTA_CONFIANZA = (0, 200, 255)  # amarillo/naranja
COLOR_DESCONOCIDO = (0, 0, 220)       # rojo


@dataclass
class ResultadoReconocimiento:
    """Lo que devuelve el reconocedor para un rostro."""

    persona_id: Optional[int]
    nombre: Optional[str]
    similitud: float          # puntuacion agregada de la mejor persona
    identificado: bool        # True si supero el umbral

    @property
    def alta_confianza(self):
        """True si la coincidencia es lo bastante solida como para ofrecer
        guardarla como nueva muestra validada."""
        return self.identificado and self.similitud >= config.UMBRAL_ALTA_CONFIANZA

    @property
    def estado(self):
        return "IDENTIFICADO" if self.identificado else "DESCONOCIDO"

    @property
    def etiqueta(self):
        """Texto corto para mostrar junto al rostro."""
        if self.identificado:
            return "{} ({:.2f})".format(self.nombre, self.similitud)
        return "DESCONOCIDO ({:.2f})".format(self.similitud)

    @property
    def color(self):
        if not self.identificado:
            return COLOR_DESCONOCIDO
        if self.alta_confianza:
            return COLOR_IDENTIFICADO
        return COLOR_ALTA_CONFIANZA


class Reconocedor:
    """
    Mantiene en memoria todos los embeddings registrados y resuelve consultas.

    Los datos se cargan una vez y se refrescan con recargar() cuando se
    registra o se elimina una persona, para no ir a disco en cada frame.
    """

    def __init__(self):
        self._matriz = None
        self._identificadores = None
        self._nombres = {}
        self.recargar()

    def recargar(self):
        """Vuelve a leer los embeddings desde SQLite."""
        matriz, identificadores, nombres = base_datos.cargar_todos_los_embeddings()
        self._matriz = matriz
        self._identificadores = identificadores
        self._nombres = nombres
        logger.info(
            "Reconocedor cargado: %s muestras de %s personas",
            len(identificadores),
            len(nombres),
        )

    @property
    def hay_datos(self):
        """True si hay al menos una muestra con la que comparar."""
        return self._matriz is not None and len(self._matriz) > 0

    @property
    def total_muestras(self):
        return 0 if self._matriz is None else len(self._matriz)

    def identificar(self, embedding):
        """
        Compara un embedding contra la base y devuelve ResultadoReconocimiento.

        Si todavia no hay nadie registrado, devuelve DESCONOCIDO con
        similitud 0.
        """
        if not self.hay_datos:
            return ResultadoReconocimiento(None, None, 0.0, False)

        consulta = np.asarray(embedding, dtype=np.float32)

        # Producto punto contra todas las muestras a la vez. Como los vectores
        # estan normalizados, cada valor ES la similitud coseno.
        similitudes = self._matriz @ consulta

        mejor_id = None
        mejor_puntuacion = -1.0

        # Se agrega por persona para que una persona con muchas muestras no
        # tenga ventaja automatica sobre otra con pocas.
        for persona_id in np.unique(self._identificadores):
            suyas = similitudes[self._identificadores == persona_id]
            puntuacion = self._agregar(suyas)
            if puntuacion > mejor_puntuacion:
                mejor_puntuacion = puntuacion
                mejor_id = int(persona_id)

        identificado = mejor_puntuacion >= config.UMBRAL_RECONOCIMIENTO

        return ResultadoReconocimiento(
            persona_id=mejor_id if identificado else None,
            nombre=self._nombres.get(mejor_id) if identificado else None,
            similitud=float(mejor_puntuacion),
            identificado=identificado,
        )

    @staticmethod
    def _agregar(similitudes_persona):
        """Aplica la estrategia de agregacion configurada a las similitudes
        de UNA persona."""
        if config.ESTRATEGIA_AGREGACION == "mejor":
            return float(similitudes_persona.max())

        # "top_k_media": media de las K muestras mas parecidas.
        k = min(config.TOP_K, len(similitudes_persona))
        mejores = np.sort(similitudes_persona)[-k:]
        return float(mejores.mean())


# ---------------------------------------------------------------------------
# DIBUJO SOBRE EL VIDEO
# ---------------------------------------------------------------------------


def _texto_dibujable(texto):
    """
    Quita tildes y enes para dibujar con OpenCV.

    cv2.putText solo maneja caracteres ASCII: un nombre como 'Martin Nunez'
    con tildes saldria con simbolos raros. El nombre COMPLETO y correcto se
    muestra siempre en las etiquetas de la interfaz Tkinter; esto es solo para
    el texto superpuesto al video.
    """
    sin_tildes = unicodedata.normalize("NFKD", str(texto))
    return sin_tildes.encode("ascii", "ignore").decode("ascii")


def dibujar_resultado(frame, caja, etiqueta, color):
    """
    Dibuja el rectangulo del rostro y su etiqueta sobre el frame (in place).

    Se usa tanto en el reconocimiento como en el registro, para no duplicar
    el codigo de dibujo.
    """
    x1, y1, x2, y2 = caja
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if not etiqueta:
        return frame

    texto = _texto_dibujable(etiqueta)
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.6
    grosor = 2

    (ancho_texto, alto_texto), base = cv2.getTextSize(texto, fuente, escala, grosor)

    # La etiqueta va encima del rectangulo, salvo que no quepa: entonces
    # se dibuja justo debajo del borde superior.
    y_fondo = y1 - alto_texto - base - 4
    if y_fondo < 0:
        y_fondo = y1 + 2

    cv2.rectangle(
        frame,
        (x1, y_fondo),
        (x1 + ancho_texto + 8, y_fondo + alto_texto + base + 4),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        texto,
        (x1 + 4, y_fondo + alto_texto + 2),
        fuente,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA,
    )
    return frame
