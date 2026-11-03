# gait-cycle-manifold

Autoencoder recurrente (GRU) para el manifold de ciclos completos de marcha, con dos modificaciones del decodificador, un protocolo de evaluación con cinco semillas y un experimento con un codificador fundacional de series de tiempo congelado.

Un ciclo de marcha normalizado es $G \in \mathbb{R}^{101\times 6}$: seis ángulos articulares (cadera, rodilla y tobillo, izquierdos y derechos) remuestreados a 101 puntos de fase. El modelo aprende la variabilidad de los ciclos de un grupo control y se evalúa en tres aspectos: fidelidad en controles no vistos, recuperación de perturbaciones sintéticas y tipicidad de ciclos de casos externos. El trabajo parte del autoencoder recurrente del proyecto NeuroStrideAI y usa los ciclos que genera su pipeline de preprocesamiento.

## Resumen del estudio

El autoencoder original presentaba dos defectos: el código latente solamente entraba al decodificador como estado inicial y la pérdida no exigía que el último punto del ciclo coincidiera con el primero. Se evaluaron dos modificaciones, entregar el código en cada paso del decodificador y añadir a la pérdida un término de cierre del ciclo. Con cinco semillas y prueba t pareada, el error absoluto medio del modelo entrenado con ciclos limpios bajó de 4.99 a 4.11 grados y el salto de cierre de 5.07 a 0.87 grados. En el modelo entrenado con ciclos corruptos, el salto bajó de 3.04 a 0.57 grados y la ganancia de corrección subió de 0.296 a 0.337. Un modelo fundacional (TimesFM 2.5) usado como codificador congelado no mejoró el error frente a pesos aleatorios y quedó por encima del GRU. Las tablas completas, con desviaciones y valores p, están en results/README.md.

## Estructura

```text
src/cycle_manifold/   paquete de Python
tests/                pruebas con datos sintéticos
notebooks/            cuadernos de ejecución y análisis
docs/                 formulación matemática y protocolo experimental
results/              salidas de los experimentos y resumen de resultados
reports/              informes, guía y diapositivas (LaTeX y PDF)
data/                 ubicación de los ciclos normalizados (no versionados)
```

El paquete se organiza así. config.py define las constantes, las semillas y las cuatro variantes del modelo. data.py lee los ciclos, aplica la partición y estandariza con el conjunto de entrenamiento. corruption.py genera la corrupción sintética. models.py contiene el autoencoder y el decodificador GRU. losses.py define la pérdida de reconstrucción y la de cierre. training.py contiene el bucle de entrenamiento con parada temprana. metrics.py y evaluation.py calculan las tres evaluaciones. statistics.py resume por semillas y aplica la prueba pareada. experiments.py ejecuta el protocolo de cinco semillas. foundation.py y foundation_suite.py implementan el codificador fundacional, sus controles y la selección de configuración. cli.py expone la línea de comandos.

## Instalación

Se requiere Python 3.10 o superior.

```bash
pip install -e ".[dev,notebooks]"
pip install -e ".[foundation]"      # solo para el experimento con TimesFM
```

## Datos

El repositorio no incluye los datos. El archivo de ciclos debe colocarse en data/cycles_normalized.parquet; el formato y el origen se describen en data/README.md.

## Uso

Comparación del GRU original y el refinado con cinco semillas (aproximadamente 15 minutos en CPU):

```bash
python -m cycle_manifold gru --out results/gru_refinement
```

Una sola variante y una sola semilla, como verificación rápida (debe imprimir un MAE de 4.038 grados):

```bash
python -m cycle_manifold gru --specs A_refined --seeds 42
```

Experimento con el codificador fundacional (descarga los pesos de TimesFM desde Hugging Face; se recomienda GPU):

```bash
python -m cycle_manifold foundation-suite --out results/foundation
```

Validación del flujo sin descargar el modelo, con pesos aleatorios y pocas épocas:

```bash
python -m cycle_manifold foundation-suite --dry-run --seeds 42 --epochs 2 --out /tmp/fm_dry
```

La opción --data permite indicar otro archivo de ciclos, y la variable CYCLE_MANIFOLD_DATA fija la ruta por defecto.

## Cuadernos

El cuaderno 01 compara el autoencoder original y el refinado, con pruebas pareadas y figuras. El cuaderno 02 documenta el experimento con TimesFM y muestra los resultados guardados; la parte que descarga los pesos preentrenados requiere acceso a Hugging Face y, idealmente, GPU.

## Pruebas

```bash
pytest
```

Las pruebas usan datos sintéticos. Verifican, entre otros aspectos, que la corrupción vectorizada coincide bit a bit con la versión por ciclo, que el entrenamiento es reproducible con la misma semilla, que la pérdida de cierre se anula en una señal periódica y que el codificador aleatorio de TimesFM tiene bloques activos.

## Reproducibilidad

Con las mismas semillas y la misma versión de PyTorch en CPU, la semilla 42 produce un MAE de 4.789 grados para el modelo A original, 4.038 para el A refinado, 5.250 para el B original y 4.739 para el B refinado. En GPU las cifras pueden diferir en el último decimal. El protocolo completo se describe en docs/protocolo.md y la formulación en docs/formulacion.md.

## Informes

El directorio reports contiene el informe final, la guía de experimentos, las diapositivas y el informe de métodos, con sus fuentes en LaTeX. Se compilan con pdflatex.

## Limitaciones

Los hiperparámetros de las variantes recurrentes se eligieron observando el conjunto de prueba de control, por lo que las cifras finales son algo optimistas. Cada ensamble proviene de un único conjunto de cinco redes. La evaluación II usa una realización fija del daño sintético. La configuración del codificador fundacional se eligió con una sola semilla, y el estudio cubre un solo modelo fundacional. Ningún valor de las métricas debe interpretarse como severidad clínica validada.
