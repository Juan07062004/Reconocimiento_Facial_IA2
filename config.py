# -*- coding: utf-8 -*-
"""
config.py
---------
Configuracion central del proyecto de reconocimiento facial.

Aqui viven TODAS las rutas y los parametros ajustables. La idea es que para
calibrar el sistema (sobre todo el umbral de reconocimiento) solo haya que
tocar este archivo y nunca la logica.

Importante para la portabilidad: todas las rutas se construyen a partir de la
carpeta donde vive este archivo, por lo que el proyecto se puede copiar a
cualquier computador sin modificar nada.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS (siempre relativas al proyecto, nunca absolutas tipo C:\Users\...)
# ---------------------------------------------------------------------------

# Carpeta que contiene este archivo = raiz del proyecto.
DIRECTORIO_BASE = Path(__file__).resolve().parent

DIRECTORIO_BD = DIRECTORIO_BASE / "database"
RUTA_BD = DIRECTORIO_BD / "reconocimiento.db"

DIRECTORIO_ROSTROS = DIRECTORIO_BASE / "faces"
DIRECTORIO_LOGS = DIRECTORIO_BASE / "logs"

# Los modelos preentrenados de InsightFace se descargan aqui (dentro del
# proyecto) para que la carpeta sea autocontenida y facil de transportar.
DIRECTORIO_MODELOS = DIRECTORIO_BASE / "modelos"

RUTA_LOG = DIRECTORIO_LOGS / "aplicacion.log"

# ---------------------------------------------------------------------------
# CAMARA
# ---------------------------------------------------------------------------

# Indice de la camara del sistema. Si hay varias camaras conectadas puede ser
# necesario probar con 1, 2, etc.
INDICE_CAMARA = 0

ANCHO_CAMARA = 640
ALTO_CAMARA = 480

# ---------------------------------------------------------------------------
# MODELO FACIAL (InsightFace + ONNX Runtime)
# ---------------------------------------------------------------------------

# 'buffalo_l' es el paquete de modelos preentrenados por defecto de InsightFace.
# Incluye detector (SCRFD) y extractor de embeddings (ArcFace, 512 dimensiones).
# NO se entrena nada: solo se usa el modelo ya entrenado.
NOMBRE_MODELO = "buffalo_l"

# Tamano al que se redimensiona la imagen para detectar. Mas grande = mas
# preciso pero mas lento. 640x640 es un buen equilibrio en CPU.
TAMANO_DETECCION = (640, 640)

# Ejecucion en CPU. Si algun dia se dispone de GPU NVIDIA con onnxruntime-gpu
# se puede anteponer "CUDAExecutionProvider".
PROVEEDORES_ONNX = ["CPUExecutionProvider"]

# Dimension del embedding que produce el modelo (ArcFace = 512).
DIMENSION_EMBEDDING = 512

# ---------------------------------------------------------------------------
# UMBRALES DE RECONOCIMIENTO
# ---------------------------------------------------------------------------
# La similitud usada es la similitud coseno entre embeddings normalizados:
# va de -1 (opuestos) a 1 (identicos).
#
# ATENCION: estos valores NO son una verdad universal. Son un punto de partida
# razonable para el modelo buffalo_l. Se DEBEN calibrar con personas conocidas
# y desconocidas usando el script pruebas/calibrar_umbral.py.
# ---------------------------------------------------------------------------

# Por debajo de este valor el resultado se considera DESCONOCIDO.
UMBRAL_RECONOCIMIENTO = 0.45

# A partir de este valor la coincidencia se considera de alta confianza y la
# interfaz ofrece guardar el rostro como nueva muestra validada.
UMBRAL_ALTA_CONFIANZA = 0.60

# Estrategia para comparar contra las VARIAS muestras de una persona:
#   "mejor"       -> se queda con la muestra mas parecida (max).
#   "top_k_media" -> promedia las TOP_K muestras mas parecidas de esa persona.
#
# "top_k_media" suele ser mas robusto porque una sola foto afortunada (o una
# mala) no decide por si sola el resultado. Si la persona tiene menos de TOP_K
# muestras se promedian todas las que tenga.
ESTRATEGIA_AGREGACION = "top_k_media"
TOP_K = 3

# ---------------------------------------------------------------------------
# CONTROL DE CALIDAD DE LAS CAPTURAS
# ---------------------------------------------------------------------------

# Tamano minimo del rostro detectado en pixeles. Rostros muy pequenos producen
# embeddings pobres.
MINIMO_ANCHO_ROSTRO = 80
MINIMO_ALTO_ROSTRO = 80

# Nitidez minima medida como varianza del Laplaciano sobre el recorte del
# rostro (medicion normalizada a 300 px de ancho: ver motor_facial.py).
# Sirve para descartar imagenes borrosas o movidas.
# 25.0: recalibrado al flujo mixto camara + fotos de celular en alta
# resolucion (una webcam borrosa da ~2.5, una foto nitida de celular da
# ~27-80; el antiguo 40.0 rechazaba fotos de celular perfectas).
MINIMA_NITIDEZ = 25.0

# Confianza minima del detector para aceptar un rostro.
MINIMA_CONFIANZA_DETECCION = 0.60

# ---------------------------------------------------------------------------
# REGISTRO DE PERSONAS
# ---------------------------------------------------------------------------

MINIMO_CAPTURAS = 5
MAXIMO_CAPTURAS = 10

# Dos capturas casi identicas (mismo gesto, mismo angulo) no aportan nada al
# perfil. Si una captura nueva se parece a otra ya tomada por encima de este
# valor, se rechaza y se pide al usuario que cambie de pose.
MAXIMA_SIMILITUD_ENTRE_CAPTURAS = 0.97

# Indicaciones que se le muestran al usuario para que varie la pose y la
# expresion. Se recorren de forma ciclica segun la captura en curso.
INDICACIONES_POSE = [
    "Mira de frente a la camara",
    "Gira ligeramente la cabeza a la izquierda",
    "Gira ligeramente la cabeza a la derecha",
    "Sonrie mirando de frente",
    "Sube un poco la barbilla",
    "Baja un poco la barbilla",
    "Expresion neutra, de frente",
    "Acercate un poco a la camara",
    "Alejate un poco de la camara",
    "Mira de frente sin gestos",
]

# ---------------------------------------------------------------------------
# RECONOCIMIENTO EN TIEMPO REAL
# ---------------------------------------------------------------------------

# No se analiza cada frame: seria innecesariamente costoso. Se analiza uno de
# cada N frames y entre medias se reutiliza el ultimo resultado para dibujar.
FRAMES_ENTRE_ANALISIS = 3

# ---------------------------------------------------------------------------
# APRENDIZAJE INCREMENTAL CONTROLADO
# ---------------------------------------------------------------------------

# Limite duro de muestras por persona para que el perfil no crezca sin control.
MAXIMO_MUESTRAS_POR_PERSONA = 30

# Nombres de las fuentes posibles de un embedding (columna 'fuente').
FUENTE_REGISTRO = "registro"
FUENTE_MUESTRA_VALIDADA = "muestra_validada"

# ---------------------------------------------------------------------------
# MEJORAMIENTO DE IMAGEN (2do proyecto del temario, sesiones 17-20)
# ---------------------------------------------------------------------------
# Si un frame muy degradado (oscuro, ruidoso o pequenio) no produce ninguna
# deteccion, el motor reintenta UNA vez tras mejorarlo con el pipeline
# clasico de mejorar_imagen.py (mediana -> gaussiano -> CLAHE -> realce).
REINTENTAR_DETECCION_MEJORANDO = True

# ---------------------------------------------------------------------------
# UTILIDADES DE ARRANQUE
# ---------------------------------------------------------------------------


def crear_directorios():
    """Crea database/, faces/, logs/ y modelos/ si todavia no existen."""
    for directorio in (
        DIRECTORIO_BD,
        DIRECTORIO_ROSTROS,
        DIRECTORIO_LOGS,
        DIRECTORIO_MODELOS,
    ):
        directorio.mkdir(parents=True, exist_ok=True)


def configurar_logging(nivel=logging.INFO):
    """
    Deja el log listo para escribir en logs/aplicacion.log y en consola.

    Privacidad: en el log NUNCA se escriben fotografias ni embeddings, solo
    eventos (persona registrada, camara abierta, errores...).
    """
    crear_directorios()

    formato = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    raiz = logging.getLogger()
    raiz.setLevel(nivel)

    # Se evita duplicar manejadores si la funcion se llama mas de una vez.
    if raiz.handlers:
        return

    manejador_archivo = logging.FileHandler(RUTA_LOG, encoding="utf-8")
    manejador_archivo.setFormatter(formato)
    raiz.addHandler(manejador_archivo)

    manejador_consola = logging.StreamHandler()
    manejador_consola.setFormatter(formato)
    raiz.addHandler(manejador_consola)
