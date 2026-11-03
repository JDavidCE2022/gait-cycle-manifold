# Resultados del manifold de ciclos de marcha

Este directorio contiene las salidas de los experimentos con el autoencoder recurrente (GRU) y con el codificador fundacional congelado. Todas las cifras se obtienen con el protocolo descrito en docs/protocolo.md: partición por sujeto, cinco semillas (42, 1, 7, 123 y 2024) y estandarización con el conjunto de entrenamiento. Las medias se reportan con la desviación estándar muestral entre semillas.

## Archivos

El directorio gru_refinement contiene las salidas de python -m cycle_manifold gru: gru_per_seed.json con todas las métricas por semilla y por ensamble, gru_summary.csv con el resumen, dos archivos, gru_paired_A_refined_vs_original.csv y gru_paired_B_refined_vs_original.csv, con las pruebas t pareadas y el registro de la ejecución, gru_benchmark.log.

El directorio foundation contiene el resultado del experimento con TimesFM ejecutado en GPU (resultados_modelo_fundacional.csv y configuracion_elegida.json), y el control de pesos aleatorios corregido (control_random_active_A.json y su registro). El directorio foundation_dry_run contiene la validación del flujo completo con pesos aleatorios, dos épocas y una semilla; sus cifras no tienen interpretación científica.

## Autoencoder GRU: original frente a refinado

Las mejoras son el código latente en cada paso del decodificador y la pérdida de cierre del ciclo. El error es el MAE sobre los controles de prueba, en grados; el salto es la diferencia entre la primera y la última fase de la salida; la corrección es la ganancia de la evaluación II.

$$
\begin{array}{llcccc}
\text{Modelo} & \text{Variante} & \text{MAE} & \text{Salto} & \text{Corrección} & \text{Tipicidad} \\ \hline
\text{A} & \text{original} & 4.99 \pm 0.23 & 5.07 \pm 0.55 & 0.208 \pm 0.018 & 0.345 \pm 0.008 \\
\text{A} & \text{refinado} & 4.11 \pm 0.05 & 0.87 \pm 0.11 & 0.189 \pm 0.012 & 0.326 \pm 0.007 \\
\text{A} & \text{refinado, ensamble} & 3.91 & 0.55 & 0.229 & 0.358 \\ \hline
\text{B} & \text{original} & 5.25 \pm 0.10 & 3.04 \pm 0.31 & 0.296 \pm 0.014 & 0.383 \pm 0.004 \\
\text{B} & \text{refinado} & 4.83 \pm 0.07 & 0.57 \pm 0.05 & 0.337 \pm 0.007 & 0.414 \pm 0.008 \\
\text{B} & \text{refinado, ensamble} & 4.74 & 0.36 & 0.349 & 0.423
\end{array}
$$

En la prueba t pareada del refinado frente al original, el MAE y el salto de cierre mejoran en ambos modelos (A: $p = 0.001$ y $p = 0.0001$; B: $p = 0.0005$ y $p = 0.0001$). En el modelo B también mejoran la corrección ($p = 0.002$) y la tipicidad ($p = 0.001$). En el modelo A la corrección no cambia de forma significativa ($p = 0.077$), la tipicidad baja ($p = 0.034$) y la diferencia de pendientes en la costura sube de 0.84 a 1.17 grados ($p = 0.034$). Los valores p no se corrigen por multiplicidad.

## Codificador fundacional congelado

El codificador es TimesFM 2.5 (200 millones de parámetros) con el bloque 9 y el promedio y el último parche como agrupación, configuración elegida sobre validación con una semilla, y una proyección lineal entrenable con decaimiento de pesos $10^{-3}$. El decodificador es el refinado. Se muestran los modelos de tipo A.

$$
\begin{array}{lcccc}
\text{Codificador} & \text{MAE} & \text{Salto} & \text{Corrección} & \text{Tipicidad} \\ \hline
\text{TimesFM preentrenado} & 7.19 \pm 0.22 & 2.55 \pm 0.15 & -0.149 \pm 0.019 & 0.101 \pm 0.022 \\
\text{TimesFM, pesos aleatorios (normas activas)} & 7.23 \pm 0.48 & 2.74 \pm 0.17 & -0.017 \pm 0.042 & 0.087 \pm 0.042 \\
\text{Serie cruda (proyección lineal)} & 3.69 \pm 0.11 & 1.40 \pm 0.05 & 0.011 \pm 0.021 & 0.181 \pm 0.013 \\
\text{GRU refinado (referencia)} & 4.11 \pm 0.05 & 0.87 \pm 0.11 & 0.189 \pm 0.012 & 0.326 \pm 0.007
\end{array}
$$

El preentrenamiento no reduce el error de reconstrucción respecto de los pesos aleatorios ($p = 0.83$ en el MAE), y ambos quedan por encima del GRU. La corrección es peor con los pesos preentrenados ($p = 0.004$). Una proyección lineal de la serie cruda obtiene un MAE menor que el del codificador GRU, pero cierra peor el ciclo y tiene menor tipicidad, de modo que el codificador recurrente no limita el error de reconstrucción en este conjunto de datos. En el tipo B, TimesFM preentrenado alcanza un MAE de $10.12 \pm 0.18$ y una corrección de $-0.117 \pm 0.110$, frente a $4.83$ y $0.337$ del GRU refinado.

## Nota sobre el control de pesos aleatorios

El archivo resultados_modelo_fundacional.csv conserva la ejecución original en GPU. En ella, el control de pesos aleatorios se creó con las escalas de las normas RMS del modelo en cero, que es la inicialización por defecto de la biblioteca. Con esas escalas la salida de cada subcapa es nula y los bloques del transformador actúan como identidad, de modo que el control equivalía a un embebido aleatorio de parches sin bloques activos. En el código entregado, el codificador aleatorio fija esas escalas en uno, y el control corregido de la tabla anterior se ejecutó así. La conclusión no cambia: el MAE del control original fue $7.14 \pm 0.31$ y el del corregido, $7.23 \pm 0.48$.

## Limitaciones

Los hiperparámetros de las variantes recurrentes se eligieron observando el conjunto de prueba de control, por lo que las cifras finales son algo optimistas. Cada ensamble proviene de un único conjunto de cinco redes. La evaluación II usa una realización fija del daño sintético; con otra realización, la ganancia de corrección varía en el orden de 0.02. La configuración del codificador fundacional se eligió con una sola semilla, y el estudio cubre un solo modelo fundacional y un solo diseño de uso.
