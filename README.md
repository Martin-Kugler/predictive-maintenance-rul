# Mantenimiento Predictivo y Análisis de Fallos en Motores Industriales

Predicción de la **RUL (Remaining Useful Life)** — los ciclos de vida que le
quedan a un motor antes de fallar — a partir de datos telemáticos
(vibración, temperatura, presión), usando el dataset **NASA C-MAPSS**.

## ¿Qué resuelve este proyecto?

Un fallo inesperado en una cadena de montaje o en un aerogenerador cuesta
miles de euros por minuto de parada. En vez de esperar a que el motor falle
(mantenimiento reactivo) o revisarlo cada X horas (mantenimiento preventivo
ciego), este pipeline estima en tiempo real cuántos ciclos de vida le quedan
a cada activo, para poder programar la intervención justo antes de que falle.

## Estructura del repositorio

```
predictive-maintenance-rul/
├── generate_synthetic_data.py   # genera datos de prueba con formato C-MAPSS
├── requirements.txt
├── data/
│   └── raw/                     # aquí van train_FD001.txt, test_FD001.txt, RUL_FD001.txt
├── src/
│   ├── data/
│   │   ├── loader.py            # carga de los ficheros crudos
│   │   ├── preprocessing.py     # filtrado de ruido, cálculo de RUL, normalización
│   │   └── features.py          # features de ventana + FFT (dominio frecuencial)
│   ├── models/
│   │   ├── losses.py            # loss asimétrica custom (penaliza más los falsos negativos)
│   │   ├── baseline.py          # XGBoost con la loss custom como objetivo
│   │   └── sequence_models.py   # LSTM y 1D-CNN en PyTorch
│   ├── eval/
│   │   └── metrics.py           # RMSE, score de negocio, niveles de alerta
│   └── train.py                 # orquesta el pipeline completo
├── app/
│   └── dashboard.py             # dashboard Streamlit con alertas por motor
└── models_saved/                # artefactos entrenados (se generan al ejecutar train.py)
```

## Dataset: NASA C-MAPSS

El dataset real se descarga del **PCoE Data Set Repository** de la NASA:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Descarga `CMaps.zip`, y copia dentro de `data/raw/` al menos estos tres ficheros
del subset FD001 (motor único, una condición operativa, un modo de fallo):

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

> Si todavía no tienes el dataset descargado, puedes generar datos sintéticos
> con el mismo formato ejecutando `python generate_synthetic_data.py`. Esto te
> permite probar el pipeline entero de principio a fin.

## Instalación (usando uv): 

```bash
uv venv          
source .venv/bin/activate # en Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Cómo ejecutar el pipeline completo

```bash
# 1. (opcional, solo si no tienes el dataset real) genera datos de prueba
python generate_synthetic_data.py

# 2. entrena baseline (XGBoost) + modelo profundo (LSTM), evalúa y guarda artefactos
python -m src.train

# 3. lanza el dashboard interactivo
streamlit run app/dashboard.py
```

## Flujo técnico

**1. Preprocesado y features** (`src/data/`)
- Filtrado de ruido con media móvil por sensor y por motor.
- Cálculo de la etiqueta RUL, acotada (RUL cap) siguiendo el criterio estándar
  de la literatura de C-MAPSS: al inicio de vida del motor la degradación no
  es observable, así que no tiene sentido pedirle al modelo que la prediga.
- Features tabulares: media, desviación típica y pendiente en ventana
  deslizante + **energía espectral por bandas de frecuencia (FFT)** — esta
  última es la parte de "dominio de la frecuencia" del enunciado, y captura
  patrones vibratorios periódicos que los estadísticos simples no ven.
- Secuencias crudas normalizadas para los modelos profundos.

**2. Modelado** (`src/models/`)
- **Baseline**: XGBoost sobre las features tabulares.
- **Modelo profundo**: LSTM (y alternativa 1D-CNN incluida) en PyTorch sobre
  ventanas temporales multivariantes.
- Ambos modelos se entrenan con la **misma función de pérdida asimétrica
  custom**, para que la comparación sea justa.

**3. Métrica de negocio** (`src/models/losses.py`)
- La loss penaliza de forma exponencial y asimétrica el error `y_pred - y_true`:
  - Si el modelo **sobreestima el RUL** (dice que el motor aguanta más de lo
    que realmente aguanta → falso negativo de fallo, riesgo de parada
    catastrófica) → penalización fuerte.
  - Si el modelo **subestima el RUL** (manda a revisión antes de tiempo →
    falso positivo, solo cuesta una inspección de más) → penalización suave.
- Implementada tanto como `nn.Module` para PyTorch como objetivo custom
  (gradiente + hessiano) para XGBoost.

**4. Dashboard** (`app/dashboard.py`)
- Métricas comparativas XGBoost vs LSTM.
- Evolución del RUL real vs predicho por motor.
- Semáforo de alertas (🟢 normal / 🟡 revisar / 🔴 urgente) para toda la flota.

## Extender el proyecto

- Cambiar de subset (FD001 → FD002/FD003/FD004, con más condiciones
  operativas y modos de fallo) solo requiere pasar `subset="FD002"` al loader.
- Añadir el modelo 1D-CNN al dashboard: ya está implementado en
  `src/models/sequence_models.py`, solo falta entrenarlo en `train.py` igual
  que el LSTM y guardar sus predicciones.
- Sustituir la media móvil por un filtro Savitzky-Golay o wavelets si el
  ruido de sensor es más agresivo.
