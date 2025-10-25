# API_Transformación_Imagenes

API para aplicar transformaciones afines y filtros sobre una imagen usando operaciones matriciales (rotación, escala, traslación, cizallamiento y reflexiones). Proyecto desarrollado como examen de Graficación.

## Resumen
Esta API permite combinar transformaciones afines y aplicar filtros (ej. Sobel) sobre una imagen base. Proporciona una interfaz web interactiva.

## Demo en línea
https://api-transformacion-imagenes.onrender.com

## Requisitos
- Python 3.8+
- virtualenv (recomendado)
```bash
pip install virtualenv
```

## Instalación y ejecución local
1. Clona el repositorio:
```bash
git clone https://github.com/IsmaelJrDev/API_Transformacion_Imagenes.git
cd API_Transformacion_Imagenes
```
2. (Opcional) crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate
```
3. Instalar dependencias:
```bash
pip install -r requirements.txt
```
4. Ejecutar en modo desarrollo:
```bash
python app.py
```
La aplicación quedará accesible en http://127.0.0.1:5000

Para deploy en producción (Gunicorn):
```bash
gunicorn app:app
```

## Estructura del repositorio
- `app.py` — servidor Flask y lógica principal de transformación.
- `templates/index.html` — interfaz web con controles.
- `static/` — activos públicos (imagen base: `static/Stark.jpg`).
- `requirements.txt` — dependencias.
- `Procfile` — configuración para deploy en plataformas como Render/Heroku.
- `README.md` — archivo de presentación.

## API (resumen de endpoints)
1) POST `/api/process`  
Procesa la imagen con parámetros JSON y devuelve la imagen resultante (data URI) y la matriz usada.

- Payload ejemplo:
```json
{
  "rotation": 45,
  "scale": 1.0,
  "translateX": 10,
  "translateY": 0,
  "shearX": 0.0,
  "shearY": 0.0,
  "mirror": false,
  "flip": false,
  "toggleSobel": true
}
```

- Respuesta (JSON):
```json
{
  "image_data": "data:image/jpeg;base64,...",
  "width": 800,
  "height": 600,
  "matrix": [a, b, e, c, d, f]
}
```
Donde `matrix` es la tupla en formato PIL `(a, b, e, c, d, f)` que representa la transformación afín.

Ejemplo curl:
```bash
curl -s -X POST http://127.0.0.1:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"rotation":30,"scale":1,"translateX":0,"translateY":0,"shearX":0,"shearY":0,"mirror":false,"flip":false,"toggleSobel":false}' \
  | jq .
```

2) POST `/api/matrix_example`  
Devuelve matrices de ejemplo (representaciones PIL / lineales) y sus descripciones para referencia.

## Notas técnicas
- La función principal de combinación y aplicación de la transformación es `apply_full_transformation` en `app.py`.
- Filtros y conversión a escala de grises están implementados en `app.py` (incluye filtro Sobel).
- La UI interactiva está en `templates/index.html` y consume `/api/process`.

## Deploy
- Usar `Procfile` y Gunicorn para despliegues en Render o Heroku:
```bash
gunicorn app:app
```

---

Archivos relevantes:
- `app.py`
- `templates/index.html`
- `requirements.txt`
- `Procfile`
- `README.md`
