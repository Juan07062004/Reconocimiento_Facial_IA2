# -*- coding: utf-8 -*-
"""
fotos.py
--------
Registro y reconocimiento A PARTIR DE UNA FOTO JPG CARGADA (de a una foto).

Flujo pensado (nuevo apartado de la interfaz):

  1. El usuario elige un .jpg cuyo NOMBRE DE ARCHIVO es el de la persona
     (ejemplo: 'juan_perez.jpg' -> 'juan perez').
  2. Se detecta el rostro (si hay varios, se usa el mas grande y se avisa).
  3. Se valida la calidad con los mismos umbrales de la camara.
  4. Se identifica contra la base de datos:
       - RECONOCIDO  -> se ofrece ANEXAR la foto a la carpeta de esa persona
                        como nueva muestra validada (con confirmacion).
       - DESCONOCIDO -> se ofrece REGISTRAR a la persona nueva usando esa
                        sola foto (con confirmacion). Si ya existe alguien
                        con ese nombre, se ofrece anexarla a su perfil.

Este modulo es LOGICA PURA (sin interfaz) para poder probarlo de forma
automatica. Las ventanas de confirmacion viven en interfaz.py.

Reglas respetadas de la especificacion: nada se guarda sin confirmacion, no
se reentrena ningun modelo, rutas relativas y fotos separadas de los datos.
"""

import logging
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import base_datos
import config

logger = logging.getLogger(__name__)


class ErrorFoto(Exception):
    """Errores previsibles del flujo de fotos (nombre repetido, foto mala...)."""


# ---------------------------------------------------------------------------
# Resultado del analisis de una foto
# ---------------------------------------------------------------------------


@dataclass
class AnalisisFoto:
    """Todo lo que la interfaz necesita saber de una foto cargada."""

    ruta: Path
    estado: str                    # error_lectura | sin_rostro | calidad | reconocido | desconocido
    motivo: str = ""               # explicacion cuando algo no sirve
    nombre_sugerido: str = ""      # deducido del nombre del archivo
    imagen = None                  # frame BGR leido (por si la interfaz lo quiere mostrar
    rostro = None                  # mejor RostroDetectado (el mas grande)
    rostros_total: int = 0
    resultado: Optional[object] = None  # ResultadoReconocimiento (si hubo comparacion)
    rostros: Optional[list] = None     # TODOS los rostros detectados (modo grupo)

    @property
    def varios_rostros(self):
        return self.rostros_total > 1


# ---------------------------------------------------------------------------
# Utilidades de nombres y archivos
# ---------------------------------------------------------------------------


def nombre_desde_archivo(ruta):
    """
    'C:/fotos/Juan_Perez.jpg' -> 'Juan Perez'.

    Guiones bajos y medios cuentan como espacios. Se conservan las mayusculas
    que trae el nombre: el nombre bonito completo vive en la base de datos.
    """
    nombre = Path(ruta).stem
    nombre = re.sub(r"[_\-]+", " ", nombre).strip()
    return nombre


def _cargar_imagen(ruta):
    """
    Lee la imagen respetando la orientacion EXIF.

    Las fotos de celular en vertical guardan la rotacion en los datos EXIF,
    que cv2.imread ignora: la foto saldria acostada y el detector no ve el
    rostro (o lo ve con muy poca confianza). Pillow si entiende EXIF, asi
    que se endereza la foto antes de pasarla a OpenCV.
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(str(ruta)) as pil:
            pil = ImageOps.exif_transpose(pil)
            rgb = np.asarray(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        # Si Pillow falla (formato raro), se intenta la lectura directa.
        return cv2.imread(str(ruta))


def id_de_nombre(nombre):
    """Busca el id de una persona por nombre (ignorando mayusculas). None si no esta."""
    with base_datos._conectar() as conexion:  # noqa: SLF001 - misma libreria interna
        fila = conexion.execute(
            "SELECT id FROM personas WHERE LOWER(nombre) = LOWER(?)",
            (nombre.strip(),),
        ).fetchone()
    return int(fila["id"]) if fila else None


def _siguiente_numero(carpeta):
    """Siguiente numero libre para nombrar la foto (01, 02...)."""
    numeros = []
    for archivo in sorted(carpeta.glob("*.jpg")):
        try:
            numeros.append(int(archivo.stem))
        except ValueError:
            continue
    return (max(numeros) + 1) if numeros else 1


def _copiar_foto_numerada(carpeta, ruta_origen):
    """
    Copia la foto ORIGINAL (byte a byte, sin re-codificar) a la carpeta de la
    persona con el siguiente numero libre. Devuelve la ruta relativa al
    proyecto, igual que hace registrar.guardar_foto.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / "{:02d}.jpg".format(_siguiente_numero(carpeta))
    shutil.copy2(str(ruta_origen), str(destino))

    if not destino.exists() or destino.stat().st_size == 0:
        destino.unlink(missing_ok=True)
        raise ErrorFoto("La copia de la fotografia salio corrupta y se descarto")

    try:
        relativa = str(destino.relative_to(config.DIRECTORIO_BASE)).replace("\\", "/")
    except ValueError:
        # La carpeta vive fuera del proyecto (solo ocurre en las pruebas
        # automaticas, que usan un directorio temporal): ruta completa.
        relativa = str(destino).replace("\\", "/")
    return relativa


def _carpeta_por_persona(persona_id, nombre):
    """Carpeta faces/<id>_<nombre>/ reutilizando la normalizacion de registrar."""
    import registrar  # import local para evitar circular al cargar modulos
    return registrar.carpeta_de_persona(persona_id, nombre)


# ---------------------------------------------------------------------------
# PASO 1: analizar una foto (detectar + validar + identificar)
# ---------------------------------------------------------------------------


def procesar_foto(motor, reconocedor, ruta_foto):
    """
    Analiza UNA foto y decide que se puede hacer con ella.

    No escribe nada en disco ni en la base de datos: solo informa. Las
    validaciones son las mismas de la camara para no bajar la calidad del
    perfil (una muestra mala contamina para siempre).
    """
    ruta = Path(ruta_foto)
    analisis = AnalisisFoto(ruta=ruta, estado="sin_rostro",
                            nombre_sugerido=nombre_desde_archivo(ruta))

    imagen = _cargar_imagen(ruta)
    if imagen is None:
        analisis.estado = "error_lectura"
        analisis.motivo = "No se pudo abrir el archivo (¿es un JPG valido?)"
        return analisis
    analisis.imagen = imagen

    rostros = motor.detectar(imagen)
    analisis.rostros_total = len(rostros)
    if not rostros:
        analisis.estado = "sin_rostro"
        analisis.motivo = "No se detecto ningun rostro en la foto"
        return analisis

    # Si hay varios rostros se toma el MAS GRANDE (area de la caja) y se
    # avisa a la interfaz: es la interpretacion natural de 'de a una foto'.
    analisis.rostros = rostros  # se guardan TODOS para el modo grupo
    rostro = max(rostros, key=lambda r: r.ancho * r.alto)
    analisis.rostro = rostro

    # -- Validaciones de calidad (mismos umbrales que la camara) ------------
    if rostro.confianza_deteccion < config.MINIMA_CONFIANZA_DETECCION:
        analisis.estado, analisis.motivo = "calidad", "Deteccion poco fiable: mejora la iluminacion"
        return analisis
    if rostro.ancho < config.MINIMO_ANCHO_ROSTRO or rostro.alto < config.MINIMO_ALTO_ROSTRO:
        analisis.estado, analisis.motivo = "calidad", "Rostro demasiado pequeno: acercate o recorta la foto"
        return analisis
    if rostro.nitidez < config.MINIMA_NITIDEZ:
        analisis.estado, analisis.motivo = "calidad", "Foto borrosa: se necesita una mas nitida"
        return analisis

    # -- Identificacion contra la base --------------------------------------
    if reconocedor is None or not reconocedor.hay_datos:
        analisis.estado = "desconocido"
        return analisis

    resultado = reconocedor.identificar(rostro.embedding)
    analisis.resultado = resultado
    analisis.estado = "reconocido" if resultado.identificado else "desconocido"
    return analisis


# ---------------------------------------------------------------------------
# PASO 2a: anexar la foto a una persona YA registrada
# ---------------------------------------------------------------------------


def anexar_muestra_foto(persona_id, ruta_origen, rostro):
    """
    Copia la foto a la carpeta de la persona y guarda su embedding como
    muestra validada (fuente='muestra_validada'). Llamar SOLO tras la
    confirmacion del usuario. Devuelve (ruta_relativa, total_muestras).
    """
    persona = base_datos.obtener_persona(persona_id)
    if persona is None:
        raise ErrorFoto("La persona ya no existe en la base de datos")

    muestras = base_datos.contar_muestras(persona_id)
    if muestras >= config.MAXIMO_MUESTRAS_POR_PERSONA:
        raise ErrorFoto(
            "'{}' ya tiene el maximo de {} muestras".format(
                persona["nombre"], config.MAXIMO_MUESTRAS_POR_PERSONA
            )
        )

    carpeta = _carpeta_por_persona(persona_id, persona["nombre"])
    ruta_relativa = _copiar_foto_numerada(carpeta, ruta_origen)
    base_datos.guardar_embedding(
        persona_id=persona_id,
        embedding=rostro.embedding,
        ruta_foto=ruta_relativa,
        calidad=rostro.calidad,
        fuente=config.FUENTE_MUESTRA_VALIDADA,
    )
    logger.info("Foto anexada a id=%s (total %s)", persona_id, muestras + 1)
    return ruta_relativa, muestras + 1


# ---------------------------------------------------------------------------
# PASO 2b: registrar una persona nueva con UNA sola foto
# ---------------------------------------------------------------------------


def registrar_desde_foto(nombre, ruta_origen, rostro):
    """
    Crea la persona, copia la foto a su carpeta y guarda el embedding.

    A diferencia del registro por camara (5-10 capturas), aqui el perfil nace
    con UNA foto: el flujo es 'de a una foto' y el perfil crece despues
    anexando mas fotos con anexar_muestra_foto(). Si algo falla, se deshace
    todo para no dejar perfiles a medias. Devuelve el persona_id.
    """
    nombre = (nombre or "").strip()
    if not nombre:
        raise ErrorFoto("El nombre del archivo esta vacio: renombra la foto")
    if len(nombre) > 60:
        raise ErrorFoto("El nombre del archivo es demasiado largo")
    if base_datos.existe_nombre(nombre):
        raise ErrorFoto(
            "Ya existe una persona llamada '{}': usa la opcion de ANEXAR".format(nombre)
        )

    persona_id = base_datos.crear_persona(nombre)
    try:
        carpeta = _carpeta_por_persona(persona_id, nombre)
        ruta_relativa = _copiar_foto_numerada(carpeta, ruta_origen)
        base_datos.guardar_embedding(
            persona_id=persona_id,
            embedding=rostro.embedding,
            ruta_foto=ruta_relativa,
            calidad=rostro.calidad,
            fuente=config.FUENTE_REGISTRO,
        )
    except Exception:
        # Rollback: eliminar persona + carpeta para no dejar basura.
        logger.exception("Fallo el registro desde foto, deshaciendo")
        persona = base_datos.obtener_persona(persona_id)
        if persona:
            carpeta = _carpeta_por_persona(persona_id, persona["nombre"])
            if carpeta.exists():
                shutil.rmtree(carpeta, ignore_errors=True)
        base_datos.eliminar_persona(persona_id)
        raise

    logger.info("Persona registrada desde foto: id=%s nombre='%s'", persona_id, nombre)
    return persona_id


# ---------------------------------------------------------------------------
# MODO GRUPO: copia anotada de la foto con cajas y nombres
# ---------------------------------------------------------------------------


def guardar_anotada(ruta_original, imagen_anotada):
    """
    Guarda la copia de la foto con las cajas y nombres dibujados en la
    carpeta resultados/ del proyecto (se crea si no existe).

    Devuelve la ruta del archivo guardado. La foto original NUNCA se toca.
    """
    carpeta = config.DIRECTORIO_BASE / "resultados"
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / (Path(ruta_original).stem + "_identificada.jpg")
    exito = cv2.imwrite(str(destino), imagen_anotada, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not exito:
        raise ErrorFoto("No se pudo guardar la foto anotada en resultados/")
    return destino
