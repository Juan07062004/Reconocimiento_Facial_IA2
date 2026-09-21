# Reconocimiento Facial Local en Python

Aplicación de escritorio para **registrar personas** con la cámara y
**reconocerlas en tiempo real**. Funciona **100 % en local**: no usa Internet
(salvo una única descarga inicial del modelo), no envía datos a ningún
servidor y no depende de servicios en la nube ni de APIs de pago.

- Registro guiado de personas con varias fotografías.
- Reconocimiento en vivo con nombre y nivel de similitud.
- Etiqueta **DESCONOCIDO** cuando la evidencia no es suficiente.
- Aprendizaje incremental **controlado**: solo se añaden muestras nuevas si tú
  las confirmas.
- Base de datos SQLite creada automáticamente.
- Todo el código propio está comentado en español.

---

## 1. Requisitos

| Requisito | Detalle |
|---|---|
| Sistema | Windows, Linux o macOS |
| Python | **3.9 – 3.12** (probado en 3.10) |
| Cámara | Webcam integrada o USB |
| Espacio | ~1.5 GB (dependencias + modelo de 281 MB) |
| Internet | Solo la **primera vez**, para descargar el modelo |

> **Python 3.13 / 3.14 todavía no**: algunas dependencias aún no publican
> versiones compatibles. Usa 3.10 o 3.12.

---

## 2. Instalación paso a paso

### 2.1 Instalar Python

Descárgalo de <https://www.python.org/downloads/>.

En Windows, durante la instalación **marca la casilla "Add Python to PATH"**.

Comprueba que quedó instalado:

```bash
python --version
```

### 2.2 Abrir una terminal en la carpeta del proyecto

```bash
cd ruta/donde/copiaste/reconocimiento_facial
```

### 2.3 Crear y activar el entorno virtual

Un entorno virtual mantiene las librerías de este proyecto separadas del resto
del computador.

**Windows (PowerShell o CMD):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sabrás que está activo porque el prompt empieza por `(.venv)`.

> Si PowerShell bloquea la activación, ejecuta una vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2.4 Instalar las dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Tarda unos minutos la primera vez.

### 2.5 Descargar el modelo

**No hay que hacer nada a mano.** La primera vez que ejecutes la aplicación,
InsightFace descarga automáticamente el paquete de modelos preentrenados
`buffalo_l` (~281 MB) dentro de la carpeta del proyecto:

```
modelos/models/buffalo_l/
```

Esa descarga es la **única** vez que se necesita Internet. A partir de ahí todo
funciona sin conexión.

> **Truco para instalar sin Internet en el PC de destino:** copia la carpeta
> `modelos/` completa desde un computador donde ya funcione. La aplicación la
> encontrará y no intentará descargar nada.

### 2.6 Ejecutar

```bash
python main.py
```

El primer arranque tarda más (descarga + carga del modelo). Los siguientes
tardan unos 8-10 segundos en cargar el modelo.

---

## 3. Cómo se usa

### Agregar persona

1. Pulsa **Agregar persona** y escribe el nombre completo.
2. Se abre la cámara. Arriba aparece la pose sugerida
   (*"Gira ligeramente la cabeza a la izquierda"*) y el progreso
   (*"Captura 3 de 5"*).
3. El botón **Capturar** solo se habilita cuando el rostro es válido. Si no lo
   es, debajo del vídeo aparece el motivo:
   - *No se detecta ningún rostro*
   - *Se detecta más de un rostro: debe haber solo uno*
   - *Rostro demasiado pequeño, acércate a la cámara*
   - *Imagen borrosa, quédate quieto un momento*
   - *Captura muy parecida a otra: cambia la pose*
4. Haz entre **5 y 10 capturas** cambiando de ángulo y expresión.
5. Pulsa **Finalizar y guardar**.

> Nada se guarda en disco hasta que pulsas *Finalizar y guardar*. Si cierras
> la ventana antes, no queda ninguna persona registrada a medias.

### Reconocer personas

1. Pulsa **Reconocer personas**.
2. Cada rostro detectado se rodea con un rectángulo y su etiqueta:
   - **Verde**: identificado con alta confianza.
   - **Amarillo**: identificado, pero por poco margen.
   - **Rojo**: `DESCONOCIDO`.
3. Debajo se muestran el nombre, el estado y la similitud exacta.
4. Con **varias personas a la vez** cada rostro se evalúa por separado y recibe
   su propia etiqueta.
5. **Detener y cerrar** libera la cámara.

### Agregar como nueva muestra (aprendizaje incremental)

Durante el reconocimiento, si la coincidencia es de **alta confianza** y hay
**un solo rostro**, se habilita el botón *Agregar como nueva muestra*. Al
pulsarlo:

1. Se vuelve a analizar el fotograma actual.
2. Se comprueba otra vez que la confianza siga siendo alta.
3. Se te pide confirmación explícita.
4. Solo entonces se guarda, con `fuente = 'muestra_validada'`.

Esto **nunca ocurre solo**. Guardar automáticamente cualquier rostro
reconocido acabaría metiendo falsos positivos en el perfil y degradando el
sistema con el tiempo. **El modelo de IA no se reentrena ni se modifica**:
únicamente se añade un embedding más a la base de datos.

### Personas registradas

Lista con id, nombre, número de muestras y fecha de registro. Desde ahí puedes:

- **Agregar muestra**: añadir muestras a alguien ya registrado (útil para
  añadir "con gafas", "con gorra", "con otra iluminación").
- **Eliminar persona**: borra la persona, sus embeddings y sus fotografías.

---

## 4. Estructura del proyecto

```
reconocimiento_facial/
├── main.py              # Punto de entrada
├── config.py            # TODAS las rutas y parámetros ajustables
├── base_datos.py        # SQLite: personas y embeddings
├── motor_facial.py      # InsightFace: detección + embeddings + calidad
├── camara.py            # OpenCV: apertura, lectura y liberación de cámara
├── registrar.py         # Registro, muestras validadas y eliminación
├── reconocer.py         # Comparación de embeddings y decisión final
├── interfaz.py          # Interfaz Tkinter
├── requirements.txt
├── README.md
├── .gitignore
├── database/
│   └── reconocimiento.db
├── faces/
│   ├── 1_carlos_perez/
│   │   ├── 01.jpg
│   │   └── 02.jpg
│   └── 2_maria_gomez/
├── modelos/             # Modelo preentrenado (se descarga solo)
├── logs/
│   └── aplicacion.log
└── pruebas/
    ├── test_basico.py       # Pruebas automáticas (sin cámara)
    ├── calibrar_umbral.py   # Herramienta de calibración del umbral
    └── probar_camara.py     # Diagnóstico rápido de la cámara
```

Las carpetas `database/`, `faces/`, `logs/` y `modelos/` se crean solas si no
existen.

### Base de datos

**personas**

| Columna | Tipo |
|---|---|
| id | INTEGER PRIMARY KEY AUTOINCREMENT |
| nombre | TEXT NOT NULL |
| fecha_registro | TEXT NOT NULL |

**embeddings**

| Columna | Tipo |
|---|---|
| id | INTEGER PRIMARY KEY AUTOINCREMENT |
| persona_id | INTEGER NOT NULL → personas(id) |
| ruta_foto | TEXT (relativa al proyecto) |
| embedding | BLOB NOT NULL (512 float32) |
| calidad | REAL |
| fecha_creacion | TEXT NOT NULL |
| fuente | TEXT (`registro` o `muestra_validada`) |

Al borrar una persona sus embeddings desaparecen automáticamente
(`ON DELETE CASCADE`).

---

## 5. Cómo funciona el reconocimiento

1. **Detección**: el modelo SCRFD localiza los rostros del fotograma.
2. **Embedding**: el modelo ArcFace convierte cada rostro en un vector de 512
   números. Dos fotos de la misma persona producen vectores que apuntan casi
   en la misma dirección.
3. **Comparación**: se calcula la **similitud coseno** (de -1 a 1) contra
   *todas* las muestras guardadas, con una sola multiplicación de matrices.
4. **Agregación por persona**: las similitudes se agrupan por persona. Con la
   estrategia por defecto (`top_k_media`) se promedian sus **3 muestras más
   parecidas**, en lugar de quedarse con la mejor. Así una única coincidencia
   afortunada no decide por sí sola, lo que reduce los falsos positivos.
5. **Umbral**: gana la persona con mayor puntuación, pero solo si supera
   `UMBRAL_RECONOCIMIENTO`. Si no lo supera → **DESCONOCIDO**.

> Es preferible decir *DESCONOCIDO* de más que confundir a dos personas.

---

## 6. Calibrar el umbral (importante)

`UMBRAL_RECONOCIMIENTO` **no es un número universal**. Depende de tu cámara, de
la iluminación del sitio y de la calidad de las muestras. El valor por defecto
(`0.45`) es un punto de partida razonable para `buffalo_l`, no una verdad
absoluta.

**Procedimiento:**

1. Registra **al menos 2 o 3 personas** con 5-10 fotos cada una.
2. Ejecuta:

   ```bash
   python pruebas/calibrar_umbral.py
   ```

3. El script compara todas las muestras entre sí y las separa en dos grupos:

   - **misma persona** → debería dar similitud alta
   - **personas distintas** → debería dar similitud baja

   Después muestra, umbral a umbral, cuántos aciertos y cuántos falsos
   positivos daría cada valor, y propone el que mejor separa los dos grupos.

4. Copia el valor sugerido a `config.py`:

   ```python
   UMBRAL_RECONOCIMIENTO = 0.50
   ```

**Cómo interpretarlo:**

| Síntoma | Causa | Solución |
|---|---|---|
| Confunde a dos personas | umbral demasiado bajo | **súbelo** (0.50, 0.55…) |
| No reconoce a nadie, todo DESCONOCIDO | umbral demasiado alto o muestras malas | **bájalo** o añade mejores muestras |
| Funciona de día pero no de noche | iluminación distinta a la del registro | añade muestras con esa iluminación |

Para comprobar cómo responde ante gente **no registrada**, pon unas fotos en
una carpeta y ejecuta:

```bash
python pruebas/calibrar_umbral.py --imagenes ruta/a/la/carpeta
```

Todas deberían salir como `DESCONOCIDO`.

### Otros parámetros de `config.py`

| Parámetro | Para qué sirve |
|---|---|
| `UMBRAL_RECONOCIMIENTO` | A partir de qué similitud se acepta una identificación |
| `UMBRAL_ALTA_CONFIANZA` | Desde cuándo se ofrece guardar una muestra nueva |
| `ESTRATEGIA_AGREGACION` | `top_k_media` (robusta) o `mejor` (sensible) |
| `TOP_K` | Cuántas muestras se promedian |
| `MINIMA_NITIDEZ` | Rechaza capturas borrosas |
| `MINIMO_ANCHO_ROSTRO` / `MINIMO_ALTO_ROSTRO` | Rechaza rostros lejanos |
| `FRAMES_ENTRE_ANALISIS` | Sube este número si el vídeo va lento |
| `INDICE_CAMARA` | Cambia de cámara si tienes varias |
| `MAXIMO_MUESTRAS_POR_PERSONA` | Evita que un perfil crezca sin control |

---

## 7. Pruebas

### Pruebas automáticas

No necesitan cámara ni modelo y usan una base de datos temporal (nunca tocan
tus datos reales):

```bash
python pruebas/test_basico.py
```

Verifican la normalización y la similitud coseno, el ciclo completo de la base
de datos, el guardado y la lectura exacta de los embeddings, el borrado en
cascada, la lógica de identificación (misma persona vs. desconocido), el
control de calidad y que todas las rutas sean relativas al proyecto.

### Prueba de cámara

```bash
python pruebas/probar_camara.py
```

Muestra qué índices de cámara responden y abre una vista previa. Si aquí no
ves imagen, el problema es de la cámara o de los permisos, no del
reconocimiento.

### Pruebas manuales mínimas

| # | Prueba | Resultado esperado |
|---|---|---|
| 1 | Registrar una persona con 5-10 muestras | Se registra y aparece en la lista |
| 2 | Reconocerla con iluminación similar | IDENTIFICADO, similitud alta |
| 3 | Reconocerla con iluminación distinta | IDENTIFICADO (si baja mucho, añade muestras con esa luz) |
| 4 | Con gafas | Suele funcionar; si falla, añade una muestra con gafas |
| 5 | Con gorra | Igual que el anterior |
| 6 | Con barba / sin barba | Igual que el anterior |
| 7 | Persona **no** registrada | **DESCONOCIDO** |
| 8 | Dos personas a la vez | Cada rostro con su propia etiqueta, de forma independiente |
| 9 | Desconectar la cámara en marcha | Mensaje de error claro y cierre ordenado |
| 10 | Eliminar una persona | Desaparece de la lista, su carpeta se borra y deja de ser reconocida |

Los casos 4, 5 y 6 son la razón de ser del botón *Agregar muestra*: si el
sistema falla con gafas, añade una muestra con gafas y volverá a acertar.

---

## 8. Portabilidad: llevarlo a otro computador

1. Copia la carpeta `reconocimiento_facial/` completa.
   - **No copies `.venv/`**: los entornos virtuales no son portables.
   - Copia `modelos/` si quieres evitar la descarga en el PC nuevo.
   - Copia `database/` y `faces/` solo si quieres llevarte las personas ya
     registradas (ver el aviso de privacidad más abajo).
2. Repite los pasos 2.3, 2.4 y 2.6 de este README.

En el código **no hay ninguna ruta absoluta**: todas se construyen a partir de
la carpeta donde está `config.py`, así que da igual dónde se copie el proyecto.

---

## 9. Privacidad y seguridad

> Las fotografías de rostros son **datos biométricos** y en muchos países están
> especialmente protegidos por la ley.

- Usa la aplicación **solo con el consentimiento** de las personas registradas.
- **Nunca subas `faces/` ni `database/` a un repositorio público.** Ya están
  incluidas en `.gitignore` junto con `logs/` y `modelos/`.
- En los logs **no** se escriben fotografías ni embeddings, solo eventos
  (persona registrada, cámara abierta, errores).
- Los datos no salen del computador en ningún momento.

---

## 10. Solución de problemas

| Problema | Solución |
|---|---|
| `No se pudo abrir la cámara` | Cierra Zoom / Teams / Meet / el navegador. En Windows: *Configuración → Privacidad → Cámara*. Prueba a cambiar `INDICE_CAMARA` en `config.py`. Ejecuta `python pruebas/probar_camara.py` |
| `No se pudo cargar el modelo facial` | Necesitas Internet la primera vez. Si no puedes, copia la carpeta `modelos/` desde otro PC |
| `Faltan dependencias` | Activa el entorno virtual y ejecuta `pip install -r requirements.txt` |
| El vídeo va a tirones | Sube `FRAMES_ENTRE_ANALISIS` a 5 o 6, o baja `TAMANO_DETECCION` a `(480, 480)` en `config.py` |
| El botón *Capturar* nunca se activa | Lee el motivo bajo el vídeo: acércate, mejora la luz, quédate quieto o cambia de pose |
| Error de compilación al instalar `insightface` | Asegúrate de instalar la versión **1.0.1** de `requirements.txt`. La 0.7.3 exige compilador de C++ |
| Todo sale DESCONOCIDO | Ejecuta `python pruebas/calibrar_umbral.py` y ajusta el umbral |

El archivo `logs/aplicacion.log` guarda los errores con fecha y hora.

---

## 11. Licencias

- El código de este proyecto es de uso académico.
- **InsightFace** y sus modelos preentrenados (`buffalo_l`) se distribuyen
  para uso **no comercial** / de investigación.

**Antes de distribuir o usar este proyecto fuera del ámbito académico, revisa
las licencias de InsightFace y de los modelos preentrenados**, además de la
normativa de protección de datos que aplique en tu país.
