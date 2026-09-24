# -*- coding: utf-8 -*-
"""
registrar.py
------------
Registro de personas y gestion de las muestras de cada perfil.

Aqui vive todo lo que ESCRIBE datos de personas: crear el perfil, guardar las
fotografias en faces/, generar los embeddings, anadir muestras validadas y
eliminar a una persona por completo.

Decision importante: durante una sesion de registro NADA se escribe en disco
ni en la base de datos hasta que el usuario confirma al final. Asi, si cierra
la ventana a mitad, no queda una persona registrada a medias.
"""

import logging
import re
import shutil
import unicodedata
from pathlib import Path

import cv2
import numpy as np

import base_datos
import config

logger = logging.getLogger(__name__)


class ErrorRegistro(Exception):
    """Errores previsibles del registro (nombre repetido, pocas capturas...)."""


# ---------------------------------------------------------------------------
# NOMBRES Y CARPETAS
# ---------------------------------------------------------------------------


def normalizar_para_carpeta(nombre):
    """
    Convierte 'Maria Gomez' en 'maria_gomez'.

    Se quitan tildes y caracteres raros porque el nombre acaba formando parte
    de una ruta de disco, y asi la carpeta es valida en Windows, Linux y Mac
    por igual. El nombre bonito (con tildes) se conserva en la base de datos.
    """
    sin_tildes = unicodedata.normalize("NFKD", nombre)
    sin_tildes = sin_tildes.encode("ascii", "ignore").decode("ascii")
    limpio = re.sub(r"[^a-zA-Z0-9]+", "_", sin_tildes).strip("_").lower()
    return limpio or "persona"


def carpeta_de_persona(persona_id, nombre):
    """Ruta de la carpeta de fotos: faces/<id>_<nombre>/ (siempre relativa)."""
    return config.DIRECTORIO_ROSTROS / "{}_{}".format(
        persona_id, normalizar_para_carpeta(nombre)
    )


def _siguiente_numero_foto(carpeta):
    """Devuelve el siguiente numero libre para nombrar una foto (01, 02...)."""
    existentes = sorted(carpeta.glob("*.jpg"))
    numeros = []
    for archivo in existentes:
        try:
            numeros.append(int(archivo.stem))
        except ValueError:
            continue  # archivos con otro nombre: se ignoran
    return (max(numeros) + 1) if numeros else 1


def guardar_foto(carpeta, frame):
    """
    Guarda un frame como JPG dentro de la carpeta de la persona.

    Devuelve la ruta RELATIVA al proyecto (para que la base de datos siga
    siendo valida si se copia el proyecto a otro computador).
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "{:02d}.jpg".format(_siguiente_numero_foto(carpeta))

    exito = cv2.imwrite(str(ruta), frame)
    if not exito:
        raise ErrorRegistro("No se pudo guardar la fotografia en disco")

    # Verificacion extra: no dejar nunca un archivo vacio o corrupto.
    if not ruta.exists() or ruta.stat().st_size == 0:
        ruta.unlink(missing_ok=True)
        raise ErrorRegistro("La fotografia se guardo corrupta y fue descartada")

    return ruta.relative_to(config.DIRECTORIO_BASE)


# ---------------------------------------------------------------------------
# SESION DE REGISTRO DE UNA PERSONA NUEVA
# ---------------------------------------------------------------------------


class SesionRegistro:
    """
    Acumula las capturas validas de una persona nueva y las persiste al final.

    Uso tipico desde la interfaz:

        sesion = SesionRegistro("Carlos Perez", motor)
        ...por cada clic en 'Capturar':
        aceptada, mensaje, rostro = sesion.intentar_capturar(frame)
        ...cuando ya hay suficientes:
        persona_id = sesion.finalizar()
    """

    def __init__(self, nombre, motor):
        nombre = (nombre or "").strip()
        if not nombre:
            raise ErrorRegistro("El nombre no puede estar vacio")
        if len(nombre) > 60:
            raise ErrorRegistro("El nombre es demasiado largo")
        if base_datos.existe_nombre(nombre):
            raise ErrorRegistro(
                "Ya existe una persona registrada con el nombre '{}'".format(nombre)
            )

        self.nombre = nombre
        self.motor = motor
        # Cada elemento es (frame_copiado, rostro_detectado).
        self.capturas = []

    # -- Estado --------------------------------------------------------------

    @property
    def total(self):
        return len(self.capturas)

    @property
    def suficientes(self):
        """True cuando ya hay el minimo de fotos exigido."""
        return self.total >= config.MINIMO_CAPTURAS

    @property
    def completa(self):
        """True cuando se alcanzo el maximo de fotos."""
        return self.total >= config.MAXIMO_CAPTURAS

    @property
    def indicacion_actual(self):
        """Pose sugerida para la siguiente captura."""
        indicaciones = config.INDICACIONES_POSE
        return indicaciones[self.total % len(indicaciones)]

    @property
    def texto_progreso(self):
        """Texto tipo 'Captura 3 de 5' para mostrar en la interfaz."""
        if self.total < config.MINIMO_CAPTURAS:
            return "Captura {} de {}".format(
                self.total + 1, config.MINIMO_CAPTURAS
            )
        return "{} capturas (minimo alcanzado, maximo {})".format(
            self.total, config.MAXIMO_CAPTURAS
        )

    # -- Captura -------------------------------------------------------------

    def intentar_capturar(self, frame):
        """
        Valida un frame y lo guarda en memoria si sirve como muestra.

        Devuelve (aceptada, mensaje, rostro). 'mensaje' siempre trae un texto
        en espanol para mostrar al usuario, explique el exito o el rechazo.
        """
        if self.completa:
            return False, "Ya se alcanzo el maximo de capturas", None

        rostro, motivo = self.motor.obtener_rostro_para_captura(frame)
        if rostro is None:
            return False, motivo, None

        # Evitar capturas practicamente iguales entre si: no aportan
        # informacion nueva al perfil y dan una falsa sensacion de variedad.
        for _, anterior in self.capturas:
            parecido = float(np.dot(anterior.embedding, rostro.embedding))
            if parecido > config.MAXIMA_SIMILITUD_ENTRE_CAPTURAS:
                return False, "Captura muy parecida a otra: cambia la pose", None

        # Se guarda una copia porque el frame original se sobrescribe en el
        # siguiente ciclo de lectura de la camara.
        self.capturas.append((frame.copy(), rostro))
        return True, "Captura {} aceptada".format(self.total), rostro

    def descartar_ultima(self):
        """Elimina la ultima captura (por si salio mal)."""
        if self.capturas:
            self.capturas.pop()

    # -- Persistencia --------------------------------------------------------

    def finalizar(self):
        """
        Escribe la persona, sus fotos y sus embeddings. Devuelve el persona_id.

        Si algo falla a mitad se deshace todo (persona y carpeta) para no
        dejar un perfil incompleto.
        """
        if not self.suficientes:
            raise ErrorRegistro(
                "Hacen falta al menos {} capturas validas (hay {})".format(
                    config.MINIMO_CAPTURAS, self.total
                )
            )

        persona_id = base_datos.crear_persona(self.nombre)
        carpeta = carpeta_de_persona(persona_id, self.nombre)

        try:
            for frame, rostro in self.capturas:
                ruta_relativa = guardar_foto(carpeta, frame)
                base_datos.guardar_embedding(
                    persona_id=persona_id,
                    embedding=rostro.embedding,
                    ruta_foto=str(ruta_relativa).replace("\\", "/"),
                    calidad=rostro.calidad,
                    fuente=config.FUENTE_REGISTRO,
                )
        except Exception:
            # Rollback manual: sin esto quedaria una persona con 2 de 5 fotos.
            logger.exception("Fallo al guardar el registro, deshaciendo cambios")
            eliminar_persona_completa(persona_id)
            raise

        logger.info(
            "Registro completado: id=%s con %s muestras", persona_id, self.total
        )
        self.capturas.clear()
        return persona_id


# ---------------------------------------------------------------------------
# APRENDIZAJE INCREMENTAL CONTROLADO
# ---------------------------------------------------------------------------


def agregar_muestra_validada(persona_id, frame, rostro):
    """
    Anade una muestra nueva al perfil de una persona YA registrada.

    Esta funcion solo debe llamarse tras una confirmacion explicita del
    usuario: guardar automaticamente cualquier rostro reconocido acabaria
    metiendo falsos positivos en el perfil y degradando el sistema.

    No se reentrena ni se modifica el modelo de IA: unicamente se guarda un
    embedding mas en la base de datos.
    """
    persona = base_datos.obtener_persona(persona_id)
    if persona is None:
        raise ErrorRegistro("La persona ya no existe en la base de datos")

    muestras = base_datos.contar_muestras(persona_id)
    if muestras >= config.MAXIMO_MUESTRAS_POR_PERSONA:
        raise ErrorRegistro(
            "'{}' ya tiene el maximo de {} muestras. Elimina alguna o sube "
            "MAXIMO_MUESTRAS_POR_PERSONA en config.py.".format(
                persona["nombre"], config.MAXIMO_MUESTRAS_POR_PERSONA
            )
        )

    carpeta = carpeta_de_persona(persona_id, persona["nombre"])
    ruta_relativa = guardar_foto(carpeta, frame)
    base_datos.guardar_embedding(
        persona_id=persona_id,
        embedding=rostro.embedding,
        ruta_foto=str(ruta_relativa).replace("\\", "/"),
        calidad=rostro.calidad,
        fuente=config.FUENTE_MUESTRA_VALIDADA,
    )

    logger.info("Muestra validada anadida a id=%s (total %s)", persona_id, muestras + 1)
    return muestras + 1


# ---------------------------------------------------------------------------
# ELIMINACION
# ---------------------------------------------------------------------------


def eliminar_persona_completa(persona_id):
    """
    Borra a una persona de la base de datos Y sus fotografias del disco.

    Se localiza la carpeta a partir del nombre guardado, y como respaldo se
    borran tambien las rutas concretas registradas en la tabla embeddings.
    """
    persona = base_datos.obtener_persona(persona_id)
    if persona is None:
        return False

    carpeta = carpeta_de_persona(persona_id, persona["nombre"])
    rutas = base_datos.eliminar_persona(persona_id)

    if carpeta.exists():
        shutil.rmtree(carpeta, ignore_errors=True)
    else:
        # Respaldo por si la carpeta se renombro a mano.
        for ruta in rutas:
            archivo = Path(config.DIRECTORIO_BASE) / ruta
            if archivo.exists():
                archivo.unlink(missing_ok=True)

    logger.info("Persona y fotografias eliminadas: id=%s", persona_id)
    return True
