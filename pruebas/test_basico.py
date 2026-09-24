# -*- coding: utf-8 -*-
"""
pruebas/test_basico.py
----------------------
Pruebas automaticas de la logica que NO necesita camara ni modelo.

Cubren la base de datos, la estrategia de comparacion de embeddings y la
normalizacion de nombres de carpeta. Se pueden ejecutar en cualquier
computador, incluso sin webcam:

    python pruebas/test_basico.py

Las pruebas trabajan sobre una base de datos temporal, asi que NUNCA tocan los
datos reales de database/reconocimiento.db.
"""

import sys
import tempfile
from pathlib import Path

# Permite ejecutar este archivo directamente: anade la raiz del proyecto al
# path de Python para poder importar config, base_datos, etc.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

import config  # noqa: E402

# IMPORTANTE: se redirige la base de datos a un archivo temporal ANTES de
# importar el resto de modulos, para no escribir en los datos reales.
_TEMPORAL = tempfile.TemporaryDirectory()
config.RUTA_BD = Path(_TEMPORAL.name) / "pruebas.db"
config.DIRECTORIO_BD = Path(_TEMPORAL.name)

import base_datos  # noqa: E402
import motor_facial  # noqa: E402
import reconocer  # noqa: E402
import registrar  # noqa: E402


def vector_aleatorio(semilla):
    """Genera un embedding falso normalizado, reproducible."""
    generador = np.random.default_rng(semilla)
    return motor_facial.normalizar(
        generador.normal(size=config.DIMENSION_EMBEDDING).astype(np.float32)
    )


def vector_con_similitud(base, similitud, semilla):
    """
    Genera un vector cuya similitud coseno con 'base' es EXACTAMENTE la pedida.

    Simula otra fotografia de la misma cara de forma controlada: se toma una
    direccion perpendicular a 'base' y se mezcla con ella en la proporcion
    justa (cos * base + sen * perpendicular).
    """
    generador = np.random.default_rng(semilla)
    aleatorio = generador.normal(size=len(base)).astype(np.float32)

    # Se quita de 'aleatorio' todo lo que apunte en la direccion de 'base',
    # de modo que lo que queda es perpendicular a 'base'.
    perpendicular = aleatorio - np.dot(aleatorio, base) * base
    perpendicular = motor_facial.normalizar(perpendicular)

    seno = float(np.sqrt(1.0 - similitud ** 2))
    return motor_facial.normalizar(similitud * base + seno * perpendicular)


def limpiar_base_datos():
    """Deja la base de pruebas vacia para que cada prueba parta de cero."""
    base_datos.inicializar_base_datos()
    for persona in base_datos.obtener_personas():
        base_datos.eliminar_persona(persona["id"])


# ---------------------------------------------------------------------------
# PRUEBAS
# ---------------------------------------------------------------------------


def prueba_normalizacion_y_similitud():
    vector = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)
    normalizado = motor_facial.normalizar(vector)

    assert abs(np.linalg.norm(normalizado) - 1.0) < 1e-5, "La norma debe ser 1"
    assert abs(motor_facial.similitud_coseno(vector, vector) - 1.0) < 1e-5, \
        "Un vector consigo mismo debe dar similitud 1"

    opuesto = -vector
    assert motor_facial.similitud_coseno(vector, opuesto) < -0.99, \
        "Vectores opuestos deben dar similitud -1"

    # El vector cero no debe romper nada (division por cero).
    cero = np.zeros(512, dtype=np.float32)
    assert motor_facial.similitud_coseno(cero, vector) == 0.0


def prueba_nombres_de_carpeta():
    assert registrar.normalizar_para_carpeta("Maria Gomez") == "maria_gomez"
    assert registrar.normalizar_para_carpeta("Jose  Ramon") == "jose_ramon"
    # Tildes y enes se transforman para que la ruta sea valida en cualquier SO.
    assert registrar.normalizar_para_carpeta("Nunez") == "nunez"
    # Un nombre sin caracteres validos no puede producir una carpeta vacia.
    assert registrar.normalizar_para_carpeta("###") == "persona"


def prueba_ciclo_completo_base_datos():
    limpiar_base_datos()

    assert not base_datos.existe_nombre("Carlos Perez")
    persona_id = base_datos.crear_persona("Carlos Perez")
    assert base_datos.existe_nombre("carlos perez"), \
        "La busqueda de nombre debe ignorar mayusculas"

    original = vector_aleatorio(1)
    base_datos.guardar_embedding(
        persona_id, original, ruta_foto="faces/1_carlos_perez/01.jpg",
        calidad=0.9, fuente=config.FUENTE_REGISTRO,
    )
    assert base_datos.contar_muestras(persona_id) == 1

    # El embedding debe volver EXACTAMENTE igual desde el BLOB.
    matriz, identificadores, nombres = base_datos.cargar_todos_los_embeddings()
    assert matriz.shape == (1, config.DIMENSION_EMBEDDING)
    assert np.allclose(matriz[0], original), "El embedding se guardo mal"
    assert identificadores[0] == persona_id
    assert nombres[persona_id] == "Carlos Perez"

    # Al borrar la persona deben desaparecer sus embeddings (ON DELETE CASCADE).
    base_datos.eliminar_persona(persona_id)
    assert base_datos.obtener_persona(persona_id) is None
    assert base_datos.contar_muestras(persona_id) == 0

    matriz, _, _ = base_datos.cargar_todos_los_embeddings()
    assert len(matriz) == 0, "No deben quedar embeddings huerfanos"


def prueba_reconocimiento():
    limpiar_base_datos()

    # Dos personas con 4 muestras cada una. Las muestras de cada persona son
    # variaciones controladas de su vector base (similitud alta entre si),
    # que es justo lo que ocurre con varias fotos de la misma cara.
    cara_ana = vector_aleatorio(10)
    cara_luis = vector_aleatorio(20)

    id_ana = base_datos.crear_persona("Ana")
    id_luis = base_datos.crear_persona("Luis")

    for i in range(4):
        base_datos.guardar_embedding(
            id_ana, vector_con_similitud(cara_ana, 0.80, 100 + i))
        base_datos.guardar_embedding(
            id_luis, vector_con_similitud(cara_luis, 0.80, 200 + i))

    reconocedor = reconocer.Reconocedor()
    assert reconocedor.hay_datos
    assert reconocedor.total_muestras == 8

    # 1) Una foto nueva de Ana debe identificarse como Ana.
    resultado = reconocedor.identificar(vector_con_similitud(cara_ana, 0.85, 999))
    assert resultado.identificado, "Deberia reconocer a Ana"
    assert resultado.nombre == "Ana", "Confundio a Ana con otra persona"

    # 2) Una cara totalmente distinta debe salir DESCONOCIDO.
    resultado = reconocedor.identificar(vector_aleatorio(777))
    assert not resultado.identificado, (
        "Un desconocido no debe identificarse (revisa el umbral en config.py)"
    )
    assert resultado.estado == "DESCONOCIDO"
    assert resultado.nombre is None

    # 3) La agregacion 'top_k_media' nunca puede superar a 'mejor' (el maximo).
    consulta = vector_con_similitud(cara_luis, 0.85, 555)
    original_estrategia = config.ESTRATEGIA_AGREGACION
    try:
        config.ESTRATEGIA_AGREGACION = "mejor"
        con_maximo = reconocedor.identificar(consulta).similitud
        config.ESTRATEGIA_AGREGACION = "top_k_media"
        con_media = reconocedor.identificar(consulta).similitud
    finally:
        config.ESTRATEGIA_AGREGACION = original_estrategia

    assert con_media <= con_maximo + 1e-6, (
        "La media de las mejores no puede ser mayor que el maximo"
    )

    # Limpieza para no afectar a otras pruebas.
    registrar.eliminar_persona_completa(id_ana)
    registrar.eliminar_persona_completa(id_luis)


def prueba_sin_personas_registradas():
    """Con la base vacia, todo debe salir DESCONOCIDO sin lanzar errores."""
    limpiar_base_datos()
    reconocedor = reconocer.Reconocedor()
    assert not reconocedor.hay_datos

    resultado = reconocedor.identificar(vector_aleatorio(5))
    assert not resultado.identificado
    assert resultado.similitud == 0.0


def prueba_validacion_de_calidad():
    """La validacion debe rechazar 0 rostros, 2 rostros y rostros pequenos."""
    validar = motor_facial.MotorFacial.validar_rostro_unico

    rostro, motivo = validar([])
    assert rostro is None and "ningun rostro" in motivo

    grande = motor_facial.RostroDetectado(
        caja=(0, 0, 200, 200), embedding=vector_aleatorio(1),
        confianza_deteccion=0.9, nitidez=300.0, calidad=0.95,
    )
    rostro, motivo = validar([grande, grande])
    assert rostro is None and "mas de un rostro" in motivo

    pequeno = motor_facial.RostroDetectado(
        caja=(0, 0, 30, 30), embedding=vector_aleatorio(2),
        confianza_deteccion=0.9, nitidez=300.0, calidad=0.9,
    )
    rostro, motivo = validar([pequeno])
    assert rostro is None and "pequeno" in motivo

    borroso = motor_facial.RostroDetectado(
        caja=(0, 0, 200, 200), embedding=vector_aleatorio(3),
        confianza_deteccion=0.9, nitidez=1.0, calidad=0.5,
    )
    rostro, motivo = validar([borroso])
    assert rostro is None and "borrosa" in motivo

    # Un rostro correcto SI debe aceptarse.
    rostro, motivo = validar([grande])
    assert rostro is grande and motivo is None


def prueba_rutas_relativas():
    """Todas las rutas del proyecto deben colgar de la carpeta del proyecto."""
    for ruta in (config.DIRECTORIO_ROSTROS, config.DIRECTORIO_LOGS,
                 config.DIRECTORIO_MODELOS):
        assert str(ruta).startswith(str(config.DIRECTORIO_BASE)), \
            "La ruta {} se sale del proyecto y rompe la portabilidad".format(ruta)


# ---------------------------------------------------------------------------
# EJECUCION
# ---------------------------------------------------------------------------

PRUEBAS = [
    prueba_normalizacion_y_similitud,
    prueba_nombres_de_carpeta,
    prueba_ciclo_completo_base_datos,
    prueba_reconocimiento,
    prueba_sin_personas_registradas,
    prueba_validacion_de_calidad,
    prueba_rutas_relativas,
]


def main():
    fallidas = 0
    for prueba in PRUEBAS:
        nombre = prueba.__name__
        try:
            prueba()
            print("  OK    {}".format(nombre))
        except AssertionError as error:
            fallidas += 1
            print("  FALLO {}: {}".format(nombre, error))
        except Exception as error:  # noqa: BLE001
            fallidas += 1
            print("  ERROR {}: {}: {}".format(nombre, type(error).__name__, error))

    print()
    if fallidas:
        print("{} de {} pruebas fallaron".format(fallidas, len(PRUEBAS)))
        return 1
    print("Las {} pruebas pasaron correctamente".format(len(PRUEBAS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
