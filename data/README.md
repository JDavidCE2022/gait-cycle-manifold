# Datos

Este repositorio no incluye datos. Los modelos leen un archivo de ciclos de marcha normalizados con el siguiente formato.

## Formato

El archivo es un Parquet en formato largo, con una fila por ciclo y por punto de fase (101 puntos por ciclo). Las columnas son cycle_id, subject_id, group, anchor_side, phase_index, phase, las seis articulaciones (left_hip, left_knee, left_ankle, right_hip, right_knee y right_ankle, en grados) y split, que toma los valores train, validation, control_test y case_external_test. La partición se hace por sujeto.

También se admite un archivo .npz con las matrices X de forma (N, 101, 6), split y group. Puede generarse desde el Parquet con el comando de exportación:

```bash
python -m cycle_manifold --data data/cycles_normalized.parquet export --out data/cycles_array.npz
```

## Origen

Los ciclos se generan con el pipeline del proyecto NeuroStrideAI (notebook 09), que interpola cada intervalo de la segmentación original a 101 puntos con PCHIP, aplica el filtro de calidad (90 de 551 intervalos excluidos) y asigna la partición por sujeto. El resultado son 461 ciclos de 64 sujetos: 240 ciclos de entrenamiento, 44 de validación, 47 de prueba de control y 130 de casos externos.

## Ubicación

El archivo se coloca en data/cycles_normalized.parquet, que es la ruta que se busca por defecto. También puede indicarse con la opción --data o con la variable de entorno CYCLE_MANIFOLD_DATA.

```bash
export CYCLE_MANIFOLD_DATA=/ruta/a/cycles_normalized.parquet
```

El contenido de data/ está excluido del control de versiones mediante .gitignore.
