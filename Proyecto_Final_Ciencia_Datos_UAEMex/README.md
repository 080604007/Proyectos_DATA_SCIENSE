# Clasificación Supervisada en Campañas de Marketing Bancario: Comparativa entre Naive Bayes, SVM y Perceptrón Multicapa

**Autor:** Luis Fernando Cruz Villegas  
**Institución:** Centro Universitario UAEM Texcoco – Universidad Autónoma del Estado de México  
**Asignatura:** Ciencia de Datos (M. en C. Benito Samuel López Razo)

## Estructura del Repositorio
```text
├── data/
│   ├── raw/                 # Dataset Bank Marketing (UCI ID: 222)
│   └── processed/           # Particiones train/val/test serializadas
├── src/
│   ├── 01_eda_and_prep.py   # Carga/descarga, limpieza, EDA y partición estratificada
│   ├── 02_train_models.py   # Búsqueda de hiperparámetros (CV) y entrenamiento
│   └── 03_evaluate.py       # Métricas en Test, curvas ROC/PR y análisis de errores
├── results/
│   ├── figures/             # Gráficas generadas a 300 DPI para el artículo IEEE
│   └── metrics/             # Tablas comparativas en CSV y JSON
├── paper/
│   ├── figures/             # Copia de gráficas lista para Overleaf
│   └── main.tex             # Código fuente en LaTeX (formato IEEEtran a 2 columnas)
└── requirements.txt         # Dependencias exactas del entorno
```

## Instrucciones de Ejecución (Reproducibilidad)
1. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
   ```
2. Ejecutar el pipeline completo en orden:
   ```bash
   python src/01_eda_and_prep.py
   python src/02_train_models.py
   python src/03_evaluate.py
   ```
Todas las operaciones estocásticas utilizan `RANDOM_STATE = 42`.
