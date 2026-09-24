# -*- coding: utf-8 -*-
"""
mejorar_imagen.py
-----------------
Segundo proyecto del temario (sesiones 17-20): MEJORAMIENTO DE IMAGEN CON
BAJA CALIDAD, aplicado al reconocimiento facial.

Un rostro fotografiado con poca luz, poca resolucion o ruido produce un
embedding pobre y el reconocimiento falla. Este modulo "rescata" la imagen
con el pipeline clasico aprendido en las clases 3-5:

  1. FILTRO DE MEDIANA (clase 4): elimina ruido de sal y pimienta sin
     difuminar los bordes del rostro.
  2. GAUSSIANO SUAVE (clase 4): atenua el ruido gaussiano fino del sensor.
  3. CLAHE (contraste local adaptativo): ecualiza el histograma por zonas
     (clase 2 aplicada localmente) recuperando el rostro en sombras.
  4. UNSHARP MASK (clase 4/5): resta una version borrosa para REALZAR los
     bordes (blur_nitidez alto = bordes fuertes = mejor embedding).
  5. SUPER-RESOLUCION SIMPLE: si la imagen es pequena, se escala 2x con
     interpolacion cubica y se vuelve a realzar (los detectores esperan
     rostros de tamano razonable).

Uso como modulo:
    from mejorar_imagen import mejorar, analizar_calidad

Uso por consola:
    python mejorar_imagen.py entrada.jpg salida.jpg
"""

import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ANALISIS (metricas objetivas de calidad)
# ---------------------------------------------------------------------------
def analizar_calidad(imagen_bgr):
    """Devuelve un diccionario con metricas de calidad de la imagen.

    nitidez   : varianza del Laplaciano (bordes fuertes = alta). El mismo
                criterio que usa motor_facial.medir_nitidez.
    contraste : desviacion estandar de las intensidades (histograma ancho).
    brillo    : intensidad media.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    return {
        "nitidez": round(nitidez, 1),
        "contraste": round(float(gris.std()), 1),
        "brillo": round(float(gris.mean()), 1),
        "resolucion": "{}x{}".format(imagen_bgr.shape[1], imagen_bgr.shape[0]),
    }


# ---------------------------------------------------------------------------
# MEJORA (pipeline clasico clase 3-5)
# ---------------------------------------------------------------------------
def mejorar(imagen_bgr, modo="calidad", escalar_si_menor_de=320, fuerza_realce=0.8):
    """Aplica el pipeline de mejoramiento y devuelve la imagen mejorada.

    Parametros:
      escalar_si_menor_de : si el lado menor de la imagen esta por debajo de
                            este valor (pixeles), se escala 2x (super-resolucion
                            simple) porque los rostros diminutos dan embeddings
                            pobres.
      fuerza_realce       : peso del unsharp mask (0 no realza, 1 realce fuerte).
    """
    if imagen_bgr is None or imagen_bgr.size == 0:
        raise ValueError("La imagen esta vacia")

    if modo == "deteccion":
        # Variante calibrada EXPERIMENTALMENTE para ayudar al DETECTOR (SCRFD):
        # mediana mas fuerte (ruido de impulso abundante en frames chicos),
        # CLAHE suave, realce leve y siempre escala 2x. Calibrada con
        # pruebas/probar_mejora_deteccion.py: rescata frames invisibles.
        imagen = cv2.medianBlur(imagen_bgr, 5)
        imagen = cv2.GaussianBlur(imagen, (0, 0), 1.0)
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        imagen = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        borrosa = cv2.GaussianBlur(imagen, (0, 0), 2.5)
        imagen = cv2.addWeighted(imagen, 1.4, borrosa, -0.4, 0)
        imagen = cv2.resize(imagen, None, fx=2, fy=2,
                            interpolation=cv2.INTER_CUBIC)
        return np.clip(imagen, 0, 255).astype(np.uint8)

    imagen = imagen_bgr.copy()

    # 1. Mediana: ruido de impulso (sal y pimienta) fuera, bordes intactos.
    imagen = cv2.medianBlur(imagen, 3)

    # 2. Gaussiano muy suave: ruido fino del sensor.
    imagen = cv2.GaussianBlur(imagen, (0, 0), 0.8)

    # 3. CLAHE sobre el canal de luminancia (LAB): contraste local adaptativo.
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    canal_l = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(canal_l)
    imagen = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Unsharp mask: realce de bordes. imagen + fuerza * (imagen - borrosa)
    borrosa = cv2.GaussianBlur(imagen, (0, 0), 2.5)
    imagen = cv2.addWeighted(imagen, 1 + fuerza_realce, borrosa, -fuerza_realce, 0)

    # 5. Super-resolucion simple para imagenes pequenas.
    lado_menor = min(imagen.shape[:2])
    if lado_menor < escalar_si_menor_de:
        factor = 2
        imagen = cv2.resize(imagen, None, fx=factor, fy=factor,
                            interpolation=cv2.INTER_CUBIC)
        # Tras escalar, un realce leve recupera la nitidez perdida.
        borrosa = cv2.GaussianBlur(imagen, (0, 0), 1.5)
        imagen = cv2.addWeighted(imagen, 1.5, borrosa, -0.5, 0)

    return np.clip(imagen, 0, 255).astype(np.uint8)


def comparar(antes, despues):
    """Panel antes/despues con las metricas para informes y presentaciones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_antes = analizar_calidad(antes)
    m_despues = analizar_calidad(despues)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.2))
    ejes[0].imshow(cv2.cvtColor(antes, cv2.COLOR_BGR2RGB))
    ejes[0].set_title("ANTES\nnitidez {nitidez} | contraste {contraste} | {resolucion}".format(**m_antes))
    ejes[1].imshow(cv2.cvtColor(despues, cv2.COLOR_BGR2RGB))
    ejes[1].set_title("DESPUES\nnitidez {nitidez} | contraste {contraste} | {resolucion}".format(**m_despues))
    for eje in ejes:
        eje.set_xticks([]), eje.set_yticks([])
    fig.suptitle("Mejoramiento de imagen con baja calidad (clases 3-5 aplicadas)")
    fig.tight_layout()
    return fig, m_antes, m_despues


# ---------------------------------------------------------------------------
# CONSOLA
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Uso: python mejorar_imagen.py entrada.jpg salida.jpg")
        return 1
    imagen = cv2.imread(sys.argv[1])
    if imagen is None:
        print("[ERROR] No se pudo leer:", sys.argv[1])
        return 1
    print("Antes  :", analizar_calidad(imagen))
    mejorada = mejorar(imagen)
    print("Despues:", analizar_calidad(mejorada))
    cv2.imwrite(sys.argv[2], mejorada)
    print("Guardada en", sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
