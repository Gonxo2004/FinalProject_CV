# 🏸 **Eye of the Hawk: Sistema de Seguimiento de Pelota de Tenis** 🟢🔴  

### **📖 Descripción General**  
¡Bienvenido al proyecto **Eye of the Hawk**! 🦅 Este sistema, desarrollado para **Raspberry Pi**, emula las funcionalidades del famoso "Ojo de Halcón", detectando y siguiendo una pelota de tenis en tiempo real. 🎾  

El sistema no solo rastrea la pelota, sino que también evalúa su rendimiento con un **contador de puntos**, penalizando retrasos y premiando respuestas rápidas. 🕹️  

---

### **✨ Características Principales**  
✅ **Calibración de Cámara**: Configura la cámara usando un tablero de ajedrez.  
🔍 **Detección de Patrones**: Reconoce figuras geométricas como cuadrados, triángulos y círculos, además de colores específicos.  
🔑 **Validación por Secuencia**: Activa el sistema solo al introducir la secuencia correcta de patrones.  
🎯 **Seguimiento Preciso**: Identifica y rastrea la pelota con una **bounding box**.  
🏆 **Contador de Puntos**: Evalúa el rendimiento del sistema en tiempo real.

---

### **📂 Estructura del Proyecto**  
📜 **`final.py`**: Script principal que combina `tracker.py` y `security_code.py`.  
📜 **`tracker.py`**: Módulo que detecta y sigue la pelota.  
📜 **`security_code.py`**: Valida secuencias de patrones geométricos.  
📁 **`data/videos`**: Vídeos de pruebas del tracker y la validación de secuencias.  
📁 **`output_videos`**: Vídeos generados por el sistema, con métricas en tiempo real.

---

### **🚀 Instalación**  

#### **Requisitos Previos**  
🔧 **Hardware**:  
- Raspberry Pi con cámara compatible. 📷  
- Accesorios básicos (fuente de alimentación, tarjeta SD, etc.).  

💻 **Software**:  
- Python 3.11 🐍  
- Librerías necesarias:  
   ```bash
   pip install opencv-python numpy
-	Conectar la Cámara
  Configura y habilita la cámara de la Raspberry Pi:
    ```bash
    sudo raspi-config
   
## 🚀 Cómo Ejecutarlo

1. Clona este repositorio en tu Raspberry Pi:
   ```bash
   git clone https://github.com/tu_usuario/proyecto_ojo_halcon.git
   
2. Accede a la carpeta del proyecto:
   ```bash
   cd proyecto_ojo_halcon

3. Ejecuta el archivo final.py para poner en marcha todo el sistema:
  ```bash
   python final.py

---


