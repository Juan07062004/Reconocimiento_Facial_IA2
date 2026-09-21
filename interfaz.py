# -*- coding: utf-8 -*-
"""
interfaz.py
-----------
Interfaz grafica con Tkinter (incluido con Python: nada que instalar).

Idea general de la interfaz:

  Ventana principal
    |- Agregar persona        -> VentanaRegistro
    |- Reconocer personas     -> VentanaReconocimiento
    |- Personas registradas   -> VentanaPersonas
    |- Salir

CONCURRENCIA (importante)
-------------------------
Leer de la camara y ejecutar el modelo facial son operaciones lentas. Si se
hicieran en el hilo de Tkinter, la ventana se congelaria y no se podria ni
pulsar 'Detener'.

Por eso PanelCamara usa dos hilos:

  - Un hilo de trabajo que lee frames y ejecuta el modelo, dejando el ultimo
    frame ya dibujado en una variable compartida (protegida con un candado).
  - El hilo de Tkinter, que cada ~30 ms coge ese ultimo frame y lo pinta con
    after(). Tkinter NUNCA se toca desde el hilo de trabajo, porque no es
    seguro para hilos.
"""

import logging
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

import base_datos
import config
import reconocer
import registrar
from camara import Camara, ErrorCamara

logger = logging.getLogger(__name__)

# Cada cuantos milisegundos el hilo de Tkinter refresca la imagen mostrada.
MILISEGUNDOS_REFRESCO = 30


# ---------------------------------------------------------------------------
# PANEL DE CAMARA REUTILIZABLE
# ---------------------------------------------------------------------------


class PanelCamara(ttk.Frame):
    """
    Widget que muestra el video de la camara con anotaciones encima.

    Parametros:
      analizador: funcion(frame) -> (anotaciones, datos)
                  'anotaciones' es una lista de (caja, etiqueta, color) que se
                  dibuja sobre el video; 'datos' es informacion libre que se
                  entrega a al_actualizar.
      al_actualizar: funcion(datos) que se ejecuta en el hilo de Tkinter en
                  cada refresco, para actualizar etiquetas y botones.
      al_fallar: funcion(mensaje) si la camara deja de responder.
    """

    def __init__(self, maestro, analizador=None, al_actualizar=None,
                 al_fallar=None, **kwargs):
        super().__init__(maestro, **kwargs)

        self._analizador = analizador
        self._al_actualizar = al_actualizar
        self._al_fallar = al_fallar

        self._camara = Camara()
        self._hilo = None
        self._activo = threading.Event()
        self._candado = threading.Lock()

        # Estado compartido entre el hilo de trabajo y el de Tkinter.
        self._frame_crudo = None       # sin dibujos, para capturar
        self._frame_anotado = None     # con dibujos, para mostrar
        self._datos = None
        self._mensaje_error = None

        self._anotaciones = []
        self._contador_frames = 0
        self._imagen_tk = None         # hay que conservar la referencia
        self._id_refresco = None

        self.etiqueta_video = ttk.Label(self, anchor="center")
        self.etiqueta_video.pack()

    @property
    def activo(self):
        """True mientras la camara esta capturando."""
        return self._activo.is_set()

    # -- Arranque y parada ---------------------------------------------------

    def iniciar(self):
        """Abre la camara y arranca el hilo. Lanza ErrorCamara si falla."""
        if self._activo.is_set():
            return
        self._camara.abrir()
        self._activo.set()
        self._hilo = threading.Thread(target=self._bucle_camara, daemon=True)
        self._hilo.start()
        self._programar_refresco()

    def detener(self):
        """Para el hilo y libera la camara. Es seguro llamarla varias veces."""
        self._activo.clear()

        if self._id_refresco is not None:
            try:
                self.after_cancel(self._id_refresco)
            except tk.TclError:
                pass  # la ventana ya se estaba destruyendo
            self._id_refresco = None

        if self._hilo is not None and self._hilo.is_alive():
            # Se espera a que el hilo termine ANTES de liberar la camara,
            # para no cerrarla mientras se esta leyendo un frame.
            self._hilo.join(timeout=2.0)
        self._hilo = None

        self._camara.liberar()

    # -- Hilo de trabajo -----------------------------------------------------

    def _bucle_camara(self):
        """Lee frames y ejecuta el modelo. Se ejecuta FUERA del hilo Tkinter."""
        try:
            while self._activo.is_set():
                frame = self._camara.leer()
                if frame is None:
                    self._mensaje_error = (
                        "Se perdio la senal de la camara. "
                        "Comprueba que sigue conectada."
                    )
                    break

                self._contador_frames += 1
                datos = self._datos

                # No se analiza cada frame: seria mas lento sin ganar nada
                # visible. Entre analisis se reutilizan las ultimas cajas.
                if (self._analizador is not None
                        and self._contador_frames % config.FRAMES_ENTRE_ANALISIS == 0):
                    try:
                        self._anotaciones, datos = self._analizador(frame)
                    except Exception:  # noqa: BLE001 - el hilo no debe morir
                        logger.exception("Error analizando el frame")
                        self._anotaciones, datos = [], None

                anotado = frame.copy()
                for caja, etiqueta, color in self._anotaciones:
                    reconocer.dibujar_resultado(anotado, caja, etiqueta, color)

                with self._candado:
                    self._frame_crudo = frame
                    self._frame_anotado = anotado
                    self._datos = datos
        finally:
            # Pase lo que pase (error, cierre de ventana), la camara se libera.
            self._camara.liberar()

    # -- Hilo de Tkinter -----------------------------------------------------

    def _programar_refresco(self):
        self._id_refresco = self.after(MILISEGUNDOS_REFRESCO, self._refrescar)

    def _refrescar(self):
        """Pinta el ultimo frame disponible. Solo corre en el hilo Tkinter."""
        if not self._activo.is_set():
            return

        if self._mensaje_error:
            mensaje = self._mensaje_error
            self._mensaje_error = None
            self.detener()
            if self._al_fallar:
                self._al_fallar(mensaje)
            return

        with self._candado:
            frame = None if self._frame_anotado is None else self._frame_anotado.copy()
            datos = self._datos

        if frame is not None:
            # OpenCV trabaja en BGR y Pillow en RGB: hay que convertir.
            imagen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._imagen_tk = ImageTk.PhotoImage(imagen)
            self.etiqueta_video.configure(image=self._imagen_tk)

        if self._al_actualizar:
            self._al_actualizar(datos)

        self._programar_refresco()

    # -- Acceso al frame actual ----------------------------------------------

    def frame_actual(self):
        """Devuelve una copia del ultimo frame SIN dibujos, o None."""
        with self._candado:
            if self._frame_crudo is None:
                return None
            return self._frame_crudo.copy()


# ---------------------------------------------------------------------------
# BASE PARA LAS VENTANAS QUE CAPTURAN FOTOS
# ---------------------------------------------------------------------------


class VentanaCaptura(tk.Toplevel):
    """
    Base comun de 'Agregar persona' y 'Agregar muestra a una persona'.

    Las dos hacen lo mismo (mostrar la camara, validar calidad y capturar) y
    solo se diferencian en que hacen con la captura, asi que el codigo comun
    vive aqui y las subclases implementan _al_capturar() y _crear_botones().
    """

    def __init__(self, maestro, motor, titulo):
        super().__init__(maestro)
        self.title(titulo)
        self.resizable(False, False)
        self.motor = motor

        # Ultimo rostro valido detectado y motivo del rechazo (si lo hay).
        self._rostro_valido = None
        self._motivo = "Iniciando camara..."

        self._construir_interfaz()

        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.transient(maestro)
        self.grab_set()  # ventana modal: evita abrir dos camaras a la vez

        self.after(50, self._iniciar_camara)

    # -- Construccion --------------------------------------------------------

    def _construir_interfaz(self):
        contenedor = ttk.Frame(self, padding=10)
        contenedor.pack(fill="both", expand=True)

        self.etiqueta_indicacion = ttk.Label(
            contenedor, text="", font=("Segoe UI", 11, "bold")
        )
        self.etiqueta_indicacion.pack(pady=(0, 6))

        self.panel = PanelCamara(
            contenedor,
            analizador=self._analizar,
            al_actualizar=self._al_actualizar,
            al_fallar=self._al_fallar_camara,
        )
        self.panel.pack()

        self.etiqueta_estado = ttk.Label(contenedor, text="", foreground="#666666")
        self.etiqueta_estado.pack(pady=6)

        self.marco_botones = ttk.Frame(contenedor)
        self.marco_botones.pack(pady=(4, 0))
        self._crear_botones(self.marco_botones)

    def _crear_botones(self, marco):
        raise NotImplementedError

    # -- Camara --------------------------------------------------------------

    def _iniciar_camara(self):
        try:
            self.panel.iniciar()
        except ErrorCamara as error:
            messagebox.showerror("Camara no disponible", str(error), parent=self)
            self.cerrar()

    def _al_fallar_camara(self, mensaje):
        messagebox.showerror("Error de camara", mensaje, parent=self)
        self.cerrar()

    # -- Analisis (hilo de trabajo) ------------------------------------------

    def _analizar(self, frame):
        """
        Detecta y valida el rostro para capturar.

        Se detecta UNA sola vez por frame y el resultado se usa tanto para
        dibujar como para decidir si la captura es aceptable.
        """
        rostros = self.motor.detectar(frame)
        rostro, motivo = self.motor.validar_rostro_unico(rostros)

        if rostro is not None:
            anotaciones = [(rostro.caja, "Listo para capturar",
                            reconocer.COLOR_IDENTIFICADO)]
        else:
            # Se dibujan igualmente las caras detectadas, en rojo, con el
            # motivo por el que no sirven.
            anotaciones = [(r.caja, motivo, reconocer.COLOR_DESCONOCIDO)
                           for r in rostros]

        return anotaciones, (rostro, motivo)

    # -- Refresco (hilo de Tkinter) ------------------------------------------

    def _al_actualizar(self, datos):
        if datos is None:
            return
        self._rostro_valido, self._motivo = datos
        self._actualizar_estado()

    def _actualizar_estado(self):
        """Refresca textos y habilita/deshabilita el boton de captura."""
        if self._rostro_valido is not None:
            self.etiqueta_estado.configure(
                text="Rostro correcto (calidad {:.2f})".format(
                    self._rostro_valido.calidad
                ),
                foreground="#137333",
            )
            self.boton_capturar.configure(state="normal")
        else:
            self.etiqueta_estado.configure(
                text=self._motivo or "Buscando rostro...", foreground="#b00020"
            )
            self.boton_capturar.configure(state="disabled")

    # -- Captura -------------------------------------------------------------

    def _capturar(self):
        """Toma el frame actual y lo entrega a la subclase."""
        frame = self.panel.frame_actual()
        if frame is None:
            messagebox.showwarning(
                "Sin imagen", "Todavia no hay imagen de la camara", parent=self
            )
            return

        # Se vuelve a validar sobre el frame exacto que se va a guardar: el
        # que se estaba mostrando puede ser de hace unas decimas de segundo.
        rostro, motivo = self.motor.obtener_rostro_para_captura(frame)
        if rostro is None:
            self.etiqueta_estado.configure(
                text="Captura rechazada: {}".format(motivo), foreground="#b00020"
            )
            return

        self._al_capturar(frame, rostro)

    def _al_capturar(self, frame, rostro):
        raise NotImplementedError

    # -- Cierre --------------------------------------------------------------

    def cerrar(self):
        """Libera la camara y cierra la ventana."""
        self.panel.detener()
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# VENTANA: AGREGAR PERSONA
# ---------------------------------------------------------------------------


class VentanaRegistro(VentanaCaptura):
    """Registro de una persona nueva con varias fotografias."""

    def __init__(self, maestro, motor, nombre, al_terminar=None):
        self.sesion = registrar.SesionRegistro(nombre, motor)
        self.al_terminar = al_terminar
        super().__init__(maestro, motor, "Agregar persona: {}".format(nombre))
        self._actualizar_progreso()

    def _crear_botones(self, marco):
        self.boton_capturar = ttk.Button(
            marco, text="Capturar", command=self._capturar, state="disabled"
        )
        self.boton_capturar.grid(row=0, column=0, padx=4)

        self.boton_descartar = ttk.Button(
            marco, text="Descartar ultima", command=self._descartar, state="disabled"
        )
        self.boton_descartar.grid(row=0, column=1, padx=4)

        self.boton_guardar = ttk.Button(
            marco, text="Finalizar y guardar", command=self._guardar, state="disabled"
        )
        self.boton_guardar.grid(row=0, column=2, padx=4)

        ttk.Button(marco, text="Cancelar", command=self.cerrar).grid(
            row=0, column=3, padx=4
        )

    # -- Progreso ------------------------------------------------------------

    def _actualizar_progreso(self):
        self.etiqueta_indicacion.configure(
            text="{}  -  {}".format(
                self.sesion.texto_progreso, self.sesion.indicacion_actual
            )
        )
        self.boton_descartar.configure(
            state="normal" if self.sesion.total else "disabled"
        )
        self.boton_guardar.configure(
            state="normal" if self.sesion.suficientes else "disabled"
        )

    def _actualizar_estado(self):
        super()._actualizar_estado()
        # Al llegar al maximo ya no se permite capturar mas.
        if self.sesion.completa:
            self.boton_capturar.configure(state="disabled")
            self.etiqueta_estado.configure(
                text="Maximo de capturas alcanzado: pulsa 'Finalizar y guardar'",
                foreground="#137333",
            )

    # -- Acciones ------------------------------------------------------------

    def _al_capturar(self, frame, rostro):
        aceptada, mensaje, _ = self.sesion.intentar_capturar(frame)
        color = "#137333" if aceptada else "#b00020"
        self.etiqueta_estado.configure(text=mensaje, foreground=color)
        self._actualizar_progreso()

    def _descartar(self):
        self.sesion.descartar_ultima()
        self.etiqueta_estado.configure(text="Ultima captura descartada",
                                       foreground="#666666")
        self._actualizar_progreso()

    def _guardar(self):
        try:
            persona_id = self.sesion.finalizar()
        except registrar.ErrorRegistro as error:
            messagebox.showerror("No se pudo registrar", str(error), parent=self)
            return
        except Exception as error:  # noqa: BLE001 - se informa al usuario
            logger.exception("Error inesperado al registrar")
            messagebox.showerror("Error inesperado", str(error), parent=self)
            return

        messagebox.showinfo(
            "Persona registrada",
            "'{}' se registro correctamente con {} muestras.".format(
                self.sesion.nombre, base_datos.contar_muestras(persona_id)
            ),
            parent=self,
        )
        if self.al_terminar:
            self.al_terminar()
        self.cerrar()

    def cerrar(self):
        # Si hay capturas sin guardar se avisa antes de perderlas.
        if self.sesion.total and self.panel.activo:
            if not messagebox.askyesno(
                "Cancelar registro",
                "Hay {} capturas sin guardar. Se perderan. Continuar?".format(
                    self.sesion.total
                ),
                parent=self,
            ):
                return
        super().cerrar()


# ---------------------------------------------------------------------------
# VENTANA: AGREGAR MUESTRA A UNA PERSONA YA REGISTRADA
# ---------------------------------------------------------------------------


class VentanaAgregarMuestra(VentanaCaptura):
    """
    Anade muestras validadas al perfil de alguien ya registrado.

    Cada captura se guarda al momento porque el usuario ya eligio a mano de
    quien se trata: esa eleccion ES la validacion.
    """

    def __init__(self, maestro, motor, persona, al_terminar=None):
        self.persona = persona
        self.al_terminar = al_terminar
        super().__init__(
            maestro, motor, "Agregar muestra a: {}".format(persona["nombre"])
        )
        self._actualizar_titulo()

    def _crear_botones(self, marco):
        self.boton_capturar = ttk.Button(
            marco, text="Guardar muestra", command=self._capturar, state="disabled"
        )
        self.boton_capturar.grid(row=0, column=0, padx=4)
        ttk.Button(marco, text="Cerrar", command=self.cerrar).grid(
            row=0, column=1, padx=4
        )

    def _actualizar_titulo(self):
        muestras = base_datos.contar_muestras(self.persona["id"])
        self.etiqueta_indicacion.configure(
            text="{} tiene {} muestras (maximo {})".format(
                self.persona["nombre"], muestras, config.MAXIMO_MUESTRAS_POR_PERSONA
            )
        )

    def _al_capturar(self, frame, rostro):
        try:
            total = registrar.agregar_muestra_validada(
                self.persona["id"], frame, rostro
            )
        except registrar.ErrorRegistro as error:
            messagebox.showwarning("No se guardo", str(error), parent=self)
            return

        self.etiqueta_estado.configure(
            text="Muestra guardada (ahora tiene {})".format(total),
            foreground="#137333",
        )
        self._actualizar_titulo()
        if self.al_terminar:
            self.al_terminar()


# ---------------------------------------------------------------------------
# VENTANA: RECONOCIMIENTO EN TIEMPO REAL
# ---------------------------------------------------------------------------


class VentanaReconocimiento(tk.Toplevel):
    """Muestra la camara e identifica en vivo a las personas registradas."""

    def __init__(self, maestro, motor, reconocedor, al_terminar=None):
        super().__init__(maestro)
        self.title("Reconocimiento en tiempo real")
        self.resizable(False, False)

        self.motor = motor
        self.reconocedor = reconocedor
        self.al_terminar = al_terminar
        self._ultimo_resultado = None

        self._construir_interfaz()

        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.transient(maestro)
        self.grab_set()
        self.after(50, self._iniciar_camara)

    def _construir_interfaz(self):
        contenedor = ttk.Frame(self, padding=10)
        contenedor.pack(fill="both", expand=True)

        self.panel = PanelCamara(
            contenedor,
            analizador=self._analizar,
            al_actualizar=self._al_actualizar,
            al_fallar=self._al_fallar_camara,
        )
        self.panel.pack()

        info = ttk.Frame(contenedor)
        info.pack(fill="x", pady=8)

        self.etiqueta_nombre = ttk.Label(info, text="Buscando rostros...",
                                         font=("Segoe UI", 13, "bold"))
        self.etiqueta_nombre.grid(row=0, column=0, sticky="w")

        self.etiqueta_estado = ttk.Label(info, text="", font=("Segoe UI", 10))
        self.etiqueta_estado.grid(row=1, column=0, sticky="w")

        self.etiqueta_similitud = ttk.Label(info, text="", foreground="#666666")
        self.etiqueta_similitud.grid(row=2, column=0, sticky="w")

        botones = ttk.Frame(contenedor)
        botones.pack()

        self.boton_muestra = ttk.Button(
            botones,
            text="Agregar como nueva muestra",
            command=self._agregar_muestra,
            state="disabled",
        )
        self.boton_muestra.grid(row=0, column=0, padx=4)

        ttk.Button(botones, text="Detener y cerrar", command=self.cerrar).grid(
            row=0, column=1, padx=4
        )

    # -- Camara --------------------------------------------------------------

    def _iniciar_camara(self):
        if not self.reconocedor.hay_datos:
            messagebox.showinfo(
                "Sin personas registradas",
                "Todavia no hay nadie registrado: todos los rostros apareceran "
                "como DESCONOCIDO.\n\nUsa 'Agregar persona' primero.",
                parent=self,
            )
        try:
            self.panel.iniciar()
        except ErrorCamara as error:
            messagebox.showerror("Camara no disponible", str(error), parent=self)
            self.cerrar()

    def _al_fallar_camara(self, mensaje):
        messagebox.showerror("Error de camara", mensaje, parent=self)
        self.cerrar()

    # -- Analisis ------------------------------------------------------------

    def _analizar(self, frame):
        """
        Identifica TODOS los rostros del frame.

        Con dos o mas personas en pantalla cada una se evalua por separado y
        recibe su propia etiqueta; no hay 'ganador unico'.
        """
        rostros = self.motor.detectar(frame)
        anotaciones = []
        resultados = []

        for rostro in rostros:
            resultado = self.reconocedor.identificar(rostro.embedding)
            anotaciones.append((rostro.caja, resultado.etiqueta, resultado.color))
            resultados.append((rostro, resultado))

        return anotaciones, resultados

    def _al_actualizar(self, datos):
        resultados = datos or []
        self._ultimo_resultado = resultados

        if not resultados:
            self.etiqueta_nombre.configure(text="Buscando rostros...",
                                           foreground="#000000")
            self.etiqueta_estado.configure(text="")
            self.etiqueta_similitud.configure(text="")
            self.boton_muestra.configure(state="disabled")
            return

        if len(resultados) > 1:
            self.etiqueta_nombre.configure(
                text="{} rostros en pantalla".format(len(resultados)),
                foreground="#000000",
            )
            self.etiqueta_estado.configure(
                text="Cada rostro se identifica por separado (ver etiquetas)"
            )
            self.etiqueta_similitud.configure(text="")
            # Con varias caras no se ofrece guardar muestra: seria ambiguo.
            self.boton_muestra.configure(state="disabled")
            return

        _, resultado = resultados[0]
        if resultado.identificado:
            self.etiqueta_nombre.configure(text=resultado.nombre, foreground="#137333")
        else:
            self.etiqueta_nombre.configure(text="DESCONOCIDO", foreground="#b00020")

        self.etiqueta_estado.configure(text="Estado: {}".format(resultado.estado))
        self.etiqueta_similitud.configure(
            text="Similitud: {:.3f}  (umbral {:.2f})".format(
                resultado.similitud, config.UMBRAL_RECONOCIMIENTO
            )
        )

        # El aprendizaje incremental SOLO se ofrece con alta confianza.
        self.boton_muestra.configure(
            state="normal" if resultado.alta_confianza else "disabled"
        )

    # -- Aprendizaje incremental controlado ----------------------------------

    def _agregar_muestra(self):
        """
        Guarda el rostro actual como muestra adicional, previa confirmacion.

        Se vuelve a analizar el frame en el momento del clic para no guardar
        una imagen distinta de la que el usuario vio, y se exige de nuevo alta
        confianza: asi un falso positivo no puede colarse en el perfil.
        """
        frame = self.panel.frame_actual()
        if frame is None:
            return

        rostro, motivo = self.motor.obtener_rostro_para_captura(frame)
        if rostro is None:
            messagebox.showwarning("No se guardo", motivo, parent=self)
            return

        resultado = self.reconocedor.identificar(rostro.embedding)
        if not resultado.alta_confianza:
            messagebox.showwarning(
                "Confianza insuficiente",
                "La coincidencia bajo de {:.2f}. No se guarda para no "
                "contaminar el perfil.".format(config.UMBRAL_ALTA_CONFIANZA),
                parent=self,
            )
            return

        if not messagebox.askyesno(
            "Confirmar",
            "Guardar este rostro como nueva muestra de '{}'?\n\n"
            "Similitud: {:.3f}".format(resultado.nombre, resultado.similitud),
            parent=self,
        ):
            return

        try:
            total = registrar.agregar_muestra_validada(
                resultado.persona_id, frame, rostro
            )
        except registrar.ErrorRegistro as error:
            messagebox.showwarning("No se guardo", str(error), parent=self)
            return

        # El reconocedor tiene los embeddings en memoria: hay que recargarlo
        # para que la muestra nueva se use de inmediato.
        self.reconocedor.recargar()
        if self.al_terminar:
            self.al_terminar()

        messagebox.showinfo(
            "Muestra guardada",
            "'{}' tiene ahora {} muestras.".format(resultado.nombre, total),
            parent=self,
        )

    # -- Cierre --------------------------------------------------------------

    def cerrar(self):
        self.panel.detener()
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# VENTANA: PERSONAS REGISTRADAS
# ---------------------------------------------------------------------------


class VentanaPersonas(tk.Toplevel):
    """Lista de personas con su numero de muestras, y acciones sobre ellas."""

    def __init__(self, maestro, motor, reconocedor):
        super().__init__(maestro)
        self.title("Personas registradas")
        self.geometry("560x360")
        self.motor = motor
        self.reconocedor = reconocedor

        self._construir_interfaz()
        self.recargar()

        self.transient(maestro)
        self.grab_set()

    def _construir_interfaz(self):
        contenedor = ttk.Frame(self, padding=10)
        contenedor.pack(fill="both", expand=True)

        columnas = ("id", "nombre", "muestras", "fecha")
        self.tabla = ttk.Treeview(
            contenedor, columns=columnas, show="headings", selectmode="browse"
        )
        for columna, titulo, ancho in (
            ("id", "ID", 50),
            ("nombre", "Nombre", 200),
            ("muestras", "Muestras", 80),
            ("fecha", "Fecha de registro", 160),
        ):
            self.tabla.heading(columna, text=titulo)
            self.tabla.column(columna, width=ancho, anchor="center")
        self.tabla.column("nombre", anchor="w")
        self.tabla.pack(fill="both", expand=True)

        self.etiqueta_resumen = ttk.Label(contenedor, text="", foreground="#666666")
        self.etiqueta_resumen.pack(pady=6)

        botones = ttk.Frame(contenedor)
        botones.pack()
        ttk.Button(botones, text="Agregar muestra",
                   command=self._agregar_muestra).grid(row=0, column=0, padx=4)
        ttk.Button(botones, text="Eliminar persona",
                   command=self._eliminar).grid(row=0, column=1, padx=4)
        ttk.Button(botones, text="Actualizar",
                   command=self.recargar).grid(row=0, column=2, padx=4)
        ttk.Button(botones, text="Cerrar",
                   command=self.cerrar).grid(row=0, column=3, padx=4)

    # -- Datos ---------------------------------------------------------------

    def recargar(self):
        """Relee la lista desde la base de datos."""
        for fila in self.tabla.get_children():
            self.tabla.delete(fila)

        personas = base_datos.obtener_personas()
        for persona in personas:
            self.tabla.insert(
                "",
                "end",
                iid=str(persona["id"]),
                values=(
                    persona["id"],
                    persona["nombre"],
                    persona["muestras"],
                    persona["fecha_registro"],
                ),
            )

        total_muestras = sum(persona["muestras"] for persona in personas)
        self.etiqueta_resumen.configure(
            text="{} personas registradas, {} muestras en total".format(
                len(personas), total_muestras
            )
        )

    def _persona_seleccionada(self):
        seleccion = self.tabla.selection()
        if not seleccion:
            messagebox.showinfo(
                "Sin seleccion", "Selecciona una persona de la lista", parent=self
            )
            return None
        return base_datos.obtener_persona(int(seleccion[0]))

    # -- Acciones ------------------------------------------------------------

    def _agregar_muestra(self):
        persona = self._persona_seleccionada()
        if persona is None:
            return

        def al_terminar():
            self.reconocedor.recargar()
            self.recargar()

        # Se suelta el grab para que la ventana de captura pueda tomarlo.
        self.grab_release()
        ventana = VentanaAgregarMuestra(self, self.motor, persona, al_terminar)
        self.wait_window(ventana)
        self.grab_set()

    def _eliminar(self):
        persona = self._persona_seleccionada()
        if persona is None:
            return

        muestras = base_datos.contar_muestras(persona["id"])
        if not messagebox.askyesno(
            "Eliminar persona",
            "Se eliminara a '{}', sus {} muestras y sus fotografias.\n\n"
            "Esta accion no se puede deshacer. Continuar?".format(
                persona["nombre"], muestras
            ),
            parent=self,
        ):
            return

        registrar.eliminar_persona_completa(persona["id"])
        self.reconocedor.recargar()
        self.recargar()
        messagebox.showinfo(
            "Eliminada", "'{}' fue eliminada".format(persona["nombre"]), parent=self
        )

    def cerrar(self):
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# VENTANA PRINCIPAL
# ---------------------------------------------------------------------------


class Aplicacion(tk.Tk):
    """Menu principal de la aplicacion."""

    def __init__(self, motor, reconocedor):
        super().__init__()
        self.title("Reconocimiento facial local")
        self.geometry("420x330")
        self.resizable(False, False)

        self.motor = motor
        self.reconocedor = reconocedor

        self._construir_interfaz()
        self._actualizar_resumen()
        self.protocol("WM_DELETE_WINDOW", self.salir)

    def _construir_interfaz(self):
        contenedor = ttk.Frame(self, padding=20)
        contenedor.pack(fill="both", expand=True)

        ttk.Label(
            contenedor,
            text="Reconocimiento facial local",
            font=("Segoe UI", 15, "bold"),
        ).pack(pady=(0, 4))
        ttk.Label(
            contenedor,
            text="Funciona sin Internet y sin servicios en la nube",
            foreground="#666666",
        ).pack(pady=(0, 16))

        for texto, accion in (
            ("Agregar persona", self.agregar_persona),
            ("Reconocer personas", self.reconocer_personas),
            ("Personas registradas", self.ver_personas),
        ):
            ttk.Button(contenedor, text=texto, command=accion, width=30).pack(pady=5)

        ttk.Separator(contenedor, orient="horizontal").pack(fill="x", pady=12)
        ttk.Button(contenedor, text="Salir", command=self.salir, width=30).pack()

        self.etiqueta_resumen = ttk.Label(contenedor, text="", foreground="#666666")
        self.etiqueta_resumen.pack(pady=(14, 0))

    def _actualizar_resumen(self):
        personas = base_datos.obtener_personas()
        self.etiqueta_resumen.configure(
            text="{} personas / {} muestras registradas".format(
                len(personas), self.reconocedor.total_muestras
            )
        )

    # -- Acciones del menu ---------------------------------------------------

    def agregar_persona(self):
        nombre = simpledialog.askstring(
            "Agregar persona",
            "Nombre completo de la persona:",
            parent=self,
        )
        if not nombre or not nombre.strip():
            return

        def al_terminar():
            self.reconocedor.recargar()
            self._actualizar_resumen()

        try:
            ventana = VentanaRegistro(self, self.motor, nombre, al_terminar)
        except registrar.ErrorRegistro as error:
            messagebox.showerror("No se puede registrar", str(error), parent=self)
            return
        self.wait_window(ventana)

    def reconocer_personas(self):
        ventana = VentanaReconocimiento(
            self, self.motor, self.reconocedor, self._actualizar_resumen
        )
        self.wait_window(ventana)
        self._actualizar_resumen()

    def ver_personas(self):
        ventana = VentanaPersonas(self, self.motor, self.reconocedor)
        self.wait_window(ventana)
        self._actualizar_resumen()

    def salir(self):
        logger.info("Cerrando la aplicacion")
        self.destroy()
