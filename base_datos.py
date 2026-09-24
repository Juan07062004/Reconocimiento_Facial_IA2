# -*- coding: utf-8 -*-
"""
base_datos.py
-------------
Capa de persistencia sobre SQLite.

Se eligio SQLite porque viene incluido con Python y no necesita instalar ni
configurar ningun servidor: basta con un archivo dentro del proyecto, lo que
mantiene todo local y portable.

Decision de diseno: cada operacion abre y cierra su propia conexion. Es un
poco menos eficiente que reutilizar una conexion global, pero evita por
completo los problemas de compartir una conexion SQLite entre el hilo de la
interfaz y el hilo de la camara.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONEXION E INICIALIZACION
# ---------------------------------------------------------------------------


@contextmanager
def _conectar():
    """
    Abre una conexion, confirma los cambios al salir y SIEMPRE la cierra.

    Ojo con un detalle poco conocido de sqlite3: usar directamente
    'with sqlite3.connect(...)' confirma la transaccion pero NO cierra la
    conexion, y el archivo se queda bloqueado. Por eso se envuelve aqui.
    """
    config.crear_directorios()
    conexion = sqlite3.connect(str(config.RUTA_BD))
    conexion.row_factory = sqlite3.Row
    # SQLite no aplica las claves foraneas salvo que se activen explicitamente.
    conexion.execute("PRAGMA foreign_keys = ON")
    try:
        yield conexion
        conexion.commit()
    except Exception:
        conexion.rollback()
        raise
    finally:
        conexion.close()


def inicializar_base_datos():
    """Crea las tablas si no existen. Es seguro llamarla en cada arranque."""
    with _conectar() as conexion:
        conexion.execute(
            """
            CREATE TABLE IF NOT EXISTS personas (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre         TEXT NOT NULL,
                fecha_registro TEXT NOT NULL
            )
            """
        )
        conexion.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id     INTEGER NOT NULL,
                ruta_foto      TEXT,
                embedding      BLOB NOT NULL,
                calidad        REAL,
                fecha_creacion TEXT NOT NULL,
                fuente         TEXT,
                FOREIGN KEY(persona_id) REFERENCES personas(id) ON DELETE CASCADE
            )
            """
        )
        # Indice para que cargar los embeddings de una persona sea rapido.
        conexion.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_persona "
            "ON embeddings(persona_id)"
        )
    logger.info("Base de datos inicializada en %s", config.RUTA_BD.name)


def _ahora():
    """Fecha y hora actual en texto, formato legible y ordenable."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# CONVERSION DE EMBEDDINGS <-> BLOB
# ---------------------------------------------------------------------------
# Un embedding es un vector de 512 numeros. Se guarda como BLOB binario
# (float32) porque ocupa poco y se reconstruye sin perdida de precision.


def embedding_a_blob(embedding):
    """Convierte un vector NumPy a bytes para guardarlo en SQLite."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_a_embedding(blob):
    """Reconstruye el vector NumPy a partir de los bytes guardados."""
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# PERSONAS
# ---------------------------------------------------------------------------


def existe_nombre(nombre):
    """Indica si ya hay una persona con ese nombre (ignorando mayusculas)."""
    with _conectar() as conexion:
        fila = conexion.execute(
            "SELECT id FROM personas WHERE LOWER(nombre) = LOWER(?)",
            (nombre.strip(),),
        ).fetchone()
    return fila is not None


def crear_persona(nombre):
    """Inserta una persona nueva y devuelve su id."""
    nombre = nombre.strip()
    with _conectar() as conexion:
        cursor = conexion.execute(
            "INSERT INTO personas (nombre, fecha_registro) VALUES (?, ?)",
            (nombre, _ahora()),
        )
        persona_id = cursor.lastrowid
    logger.info("Persona creada: id=%s", persona_id)
    return persona_id


def obtener_persona(persona_id):
    """Devuelve un diccionario con los datos de la persona, o None."""
    with _conectar() as conexion:
        fila = conexion.execute(
            "SELECT id, nombre, fecha_registro FROM personas WHERE id = ?",
            (persona_id,),
        ).fetchone()
    return dict(fila) if fila else None


def obtener_personas():
    """
    Devuelve la lista de personas registradas incluyendo cuantas muestras
    (embeddings) tiene cada una. Es lo que consume la pantalla
    'Personas registradas'.
    """
    with _conectar() as conexion:
        filas = conexion.execute(
            """
            SELECT p.id,
                   p.nombre,
                   p.fecha_registro,
                   COUNT(e.id) AS muestras
            FROM personas p
            LEFT JOIN embeddings e ON e.persona_id = p.id
            GROUP BY p.id, p.nombre, p.fecha_registro
            ORDER BY p.nombre COLLATE NOCASE
            """
        ).fetchall()
    return [dict(fila) for fila in filas]


def eliminar_persona(persona_id):
    """
    Borra la persona y, gracias a ON DELETE CASCADE, todos sus embeddings.

    Devuelve la lista de rutas de fotos que quedaron huerfanas para que quien
    llame decida borrarlas del disco (esta capa no toca archivos).
    """
    with _conectar() as conexion:
        filas = conexion.execute(
            "SELECT ruta_foto FROM embeddings WHERE persona_id = ?",
            (persona_id,),
        ).fetchall()
        rutas = [fila["ruta_foto"] for fila in filas if fila["ruta_foto"]]
        conexion.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
    logger.info("Persona eliminada: id=%s", persona_id)
    return rutas


# ---------------------------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------------------------


def guardar_embedding(persona_id, embedding, ruta_foto=None, calidad=None,
                      fuente=config.FUENTE_REGISTRO):
    """Guarda un embedding asociado a una persona y devuelve su id."""
    with _conectar() as conexion:
        cursor = conexion.execute(
            """
            INSERT INTO embeddings
                (persona_id, ruta_foto, embedding, calidad, fecha_creacion, fuente)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                persona_id,
                str(ruta_foto) if ruta_foto else None,
                embedding_a_blob(embedding),
                float(calidad) if calidad is not None else None,
                _ahora(),
                fuente,
            ),
        )
        return cursor.lastrowid


def contar_muestras(persona_id):
    """Numero de embeddings que tiene una persona."""
    with _conectar() as conexion:
        fila = conexion.execute(
            "SELECT COUNT(*) AS total FROM embeddings WHERE persona_id = ?",
            (persona_id,),
        ).fetchone()
    return fila["total"]


def cargar_todos_los_embeddings():
    """
    Carga en memoria todos los embeddings registrados.

    Devuelve una tupla (matriz, identificadores, nombres):
      - matriz: array NumPy de forma (n_muestras, 512).
      - identificadores: array NumPy con el persona_id de cada fila.
      - nombres: diccionario {persona_id: nombre}.

    Se devuelve como matriz porque asi la comparacion contra TODAS las muestras
    se resuelve con un unico producto matricial, practicamente instantaneo
    aunque haya cientos de muestras.
    """
    with _conectar() as conexion:
        filas = conexion.execute(
            """
            SELECT e.persona_id, e.embedding, p.nombre
            FROM embeddings e
            JOIN personas p ON p.id = e.persona_id
            """
        ).fetchall()

    if not filas:
        matriz_vacia = np.empty((0, config.DIMENSION_EMBEDDING), dtype=np.float32)
        return matriz_vacia, np.empty((0,), dtype=np.int64), {}

    vectores = []
    identificadores = []
    nombres = {}
    for fila in filas:
        vectores.append(blob_a_embedding(fila["embedding"]))
        identificadores.append(fila["persona_id"])
        nombres[fila["persona_id"]] = fila["nombre"]

    matriz = np.vstack(vectores).astype(np.float32)
    return matriz, np.array(identificadores, dtype=np.int64), nombres
