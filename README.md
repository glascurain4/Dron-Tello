# Código para Competencia Regional de Vehiculos Autonomos (Dron Tello)

Este repositorio contiene un conjunto de scripts en Python y configuraciones para controlar e interactuar con drones durante una competencia. El código está diseñado para proporcionar funcionalidades como detección facial, seguimiento de rutas y reconocimiento de figuras, específicamente adaptadas para operaciones autónomas de drones.

## Descripción de Archivos

### Funcionalidades Principales

- **`pruebas.py`**: Un script para probar la detección facial utilizando un clasificador Haar Cascade. Detecta y resalta rostros en el flujo de video del dron, calculando su centro y área.

- **`get_battery.py`**: Un script utilitario para verificar el estado actual de la batería del dron.

- **`test_keyboard.py`**: Una interfaz de control usando `pygame` para la navegación manual del dron. Permite a los usuarios controlar el movimiento del dron y observar su flujo de video en tiempo real.

### Módulos de Comportamiento del Dron

- **`TelloBase.py`**: Una clase base que maneja las operaciones básicas del dron, como la conexión, la gestión de flujos y el procesamiento de entradas para vuelos manuales o autónomos.

- **`TelloPath.py`**: Extiende `TelloBase` con capacidades de seguimiento de rutas, incluyendo el cálculo de ajustes de velocidad basados en la alineación del dron con una ruta.

- **`TelloFigure.py`**: Amplía `TelloPath` para incluir reconocimiento de figuras basado en máscaras de color HSV. Detecta formas predefinidas y ejecuta acciones asociadas como rotar o avanzar.

### Archivos de Soporte

- **`haarcascade_frontalface_default.xml`** y **`haarcascade_fullbody.xml`**: Modelos preentrenados para detectar rostros y cuerpos completos utilizando clasificadores Haar Cascade.

- **`gitignore.txt`**: Archivo de configuración para excluir archivos innecesarios del control de versiones.

- **`hue.png`**: Imagen de referencia utilizada para ajustar los rangos HSV durante el reconocimiento de figuras.

## Uso

1. **Instalación**:
   - Asegúrate de tener Python instalado junto con las dependencias necesarias listadas a continuación.
   - Instala las dependencias utilizando:
     ```bash
     pip install djitellopy pygame opencv-python numpy
     ```

2. **Ejecutar Scripts**:
   - Verificar el estado de la batería:
     ```bash
     python get_battery.py
     ```
   - Probar detección facial:
     ```bash
     python pruebas.py
     ```
   - Usar el teclado para control manual:
     ```bash
     python test_keyboard.py
     ```

3. **Operaciones Autónomas**:
   - Extiende las clases `TelloBase`, `TelloPath` o `TelloFigure` para implementar tareas específicas de la competencia, como navegar por rutas o reconocer formas específicas.

## Créditos

Este proyecto fue desarrollado y mantenido por **cenfraGit** para su uso en competencias de drones. Se reconoce a todos los colaboradores por su experiencia y dedicación en la creación de una base de código eficiente y funcional para obtener la victoria en la competencia regional, aunado a esto la función de este repositorio es servir como referencia y almacen para mis propias pruebas y trabajo relacionado.

