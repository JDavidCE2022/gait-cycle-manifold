# Protocolo experimental

## Datos

Los ciclos se obtienen con el notebook 09 del proyecto NeuroStrideAI, a partir de 551 intervalos de la segmentación original. Cada intervalo se interpola a 101 puntos con PCHIP y conserva seis ángulos articulares (cadera, rodilla y tobillo, izquierdos y derechos). El filtro de calidad del notebook 09 excluye 90 intervalos: 54 por saltos entre fotogramas superiores al umbral, 34 por ciclos que no cierran y 2 por duración insuficiente. Quedan 461 ciclos válidos de 64 sujetos.

## Particiones

La partición se hace por sujeto, de modo que ningún sujeto aparece en dos conjuntos. Los controles se reparten en entrenamiento, validación y prueba de control; los casos forman el conjunto externo y nunca intervienen en el ajuste.

$$
\begin{array}{lrr}
\text{Conjunto} & \text{Ciclos} & \text{Sujetos} \\ \hline
\text{entrenamiento (control)} & 240 & 28 \\
\text{validación (control)} & 44 & 6 \\
\text{prueba de control} & 47 & 7 \\
\text{casos externos} & 130 & 23
\end{array}
$$

La estandarización por canal usa únicamente la media y la desviación del conjunto de entrenamiento. Las métricas se reportan siempre en grados.

## Entrenamiento

Se usa Adam con tasa de aprendizaje $10^{-3}$, lotes de 64 ciclos, hasta 100 épocas, parada temprana con paciencia de 12 épocas sobre la pérdida de validación, recorte de gradiente a norma 1 y 48 unidades ocultas. El estado con menor pérdida de validación se restaura al finalizar.

## Semillas y ensamble

Cada configuración se entrena con las semillas 42, 1, 7, 123 y 2024. Se reportan la media y la desviación estándar muestral entre semillas y, aparte, el ensamble, definido como el promedio de las predicciones de las cinco redes. Una sola semilla puede producir diferencias de décimas de grado por azar, por lo que ninguna mejora se afirma con una sola corrida.

## Contraste estadístico

Las comparaciones entre variantes usan la prueba t pareada por semilla, con nivel $\alpha = 0.05$. No se aplica corrección por multiplicidad, y los valores p deben leerse como exploratorios cuando se contrastan varias métricas a la vez.

## Evaluaciones

La evaluación I mide la fidelidad sobre controles de prueba no vistos. La evaluación II corrompe esos mismos controles con una realización fija del daño sintético y mide cuánto del daño recupera cada modelo. La evaluación III mide, sobre los casos externos, cuánto se acerca la salida al control de entrenamiento más cercano y cuánto modifica la entrada.

## Limitaciones

Los hiperparámetros de las variantes recurrentes se eligieron observando el conjunto de prueba de control, por lo que las cifras finales son algo optimistas. Cada ensamble proviene de un único conjunto de cinco redes. La evaluación II usa una sola realización del daño aleatorio; con otra realización, la ganancia de corrección varía en el orden de 0.02. Para el codificador fundacional, la configuración se eligió con una sola semilla sobre validación.
