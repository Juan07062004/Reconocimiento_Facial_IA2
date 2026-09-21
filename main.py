# -*- coding: utf-8 -*-
"""
main.py
-------
Punto de entrada de la aplicacion.

Se ejecuta con:   python main.py

Orden de arranque:
  1. Crear carpetas (database/, faces/, logs/, modelos/) si no existen.
  2. Preparar el log.
  3. Crear las tablas de SQLite si no existen.
  4. Cargar el modelo facial preentrenado (lo mas lento del arranque).
  5. Cargar en memoria los embeddings ya registrados.
  6. Abrir la ventana principal.

Si algo falla se muestra un mensaje claro en espanol, tanto en consola como en
una ventana emergente, en lugar de un error tecnico sin explicacion.
"""

import logging
import sys
import tkinter as tk
from tkinter import messagebox


def mostrar_error(titulo, mensaje):
    """Muestra el error por consola y en una ventana, si Tkinter funciona."""
    print("\n[ERROR] {}\n{}\n".format(titulo, mensaje))
    try:
        raiz = tk.Tk()
        raiz.withdraw()
        messagebox.showerror(titulo, mensaje)
        raiz.destroy()
    except Exception:  # noqa: BLE001 - si ni Tkinter va, basta la consola
        pass


def main():
    # -- Dependencias --------------------------------------------------------
    # Se importan aqui (y no arriba) para poder dar un mensaje entendible si
    # el usuario todavia no ha instalado requirements.txt.
    try:
        import config
        import base_datos
        import interfaz
        import motor_facial
        import reconocer
    except ImportError as error:
        mostrar_error(
            "Faltan dependencias",
            "No se pudo importar una libreria necesaria:\n\n  {}\n\n"
            "Activa el entorno virtual e instala las dependencias:\n"
            "    pip install -r requirements.txt".format(error),
        )
        return 1

    # -- Preparacion del entorno local ---------------------------------------
    config.crear_directorios()
    config.configurar_logging()
    logger = logging.getLogger("main")
    logger.info("Iniciando la aplicacion")

    try:
        base_datos.inicializar_base_datos()
    except Exception as error:  # noqa: BLE001
        logger.exception("Fallo al inicializar la base de datos")
        mostrar_error(
            "Error en la base de datos",
            "No se pudo preparar la base de datos:\n\n{}".format(error),
        )
        return 1

    # -- Modelo facial -------------------------------------------------------
    print("Cargando el modelo facial, espera unos segundos...", flush=True)
    print("(la PRIMERA vez puede tardar mas porque descarga el modelo)", flush=True)
    try:
        motor = motor_facial.MotorFacial()
    except motor_facial.ErrorModeloFacial as error:
        logger.error("No se pudo cargar el modelo: %s", error)
        mostrar_error("No se pudo cargar el modelo facial", str(error))
        return 1

    # -- Datos ya registrados ------------------------------------------------
    reconocedor = reconocer.Reconocedor()
    print("Listo: {} muestras cargadas.".format(reconocedor.total_muestras), flush=True)

    # -- Interfaz ------------------------------------------------------------
    aplicacion = interfaz.Aplicacion(motor, reconocedor)
    try:
        aplicacion.mainloop()
    except KeyboardInterrupt:
        # Ctrl+C en la consola: se cierra ordenadamente.
        aplicacion.destroy()

    logger.info("Aplicacion cerrada")
    return 0


if __name__ == "__main__":
    sys.exit(main())
