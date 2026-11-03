# Formulación matemática del autoencoder recurrente refinado y del codificador fundacional

Este documento extiende la formulación del autoencoder recurrente (GRU) del proyecto NeuroStrideAI (notebook 10, sección 2.5 de su documento de formulación matemática) con las dos modificaciones del decodificador recurrente y con el codificador fundacional congelado. La notación es la misma: un ciclo normalizado es $G \in \mathbb{R}^{101\times 6}$, con fase $t_k = k/100$, $k = 0,\dots,100$, y $Z$ denota el ciclo estandarizado con la media $\mu_j$ y la desviación $\sigma_j$ del conjunto de entrenamiento.

## 1. Autoencoder recurrente original

El codificador es una GRU bidireccional sobre $Z(t_0),\dots,Z(t_{100})$. Con $h^{\rightarrow}, h^{\leftarrow} \in \mathbb{R}^{48}$ los estados finales de cada dirección,

$$
z = W_{\mathrm{enc}}\,[h^{\rightarrow}\,\Vert\,h^{\leftarrow}] + b_{\mathrm{enc}} \in \mathbb{R}^{d}, \qquad d = 16 .
$$

El decodificador recibe una codificación periódica de la fase,

$$
\varphi(t_k) = \big(\sin 2\pi t_k,\ \cos 2\pi t_k\big) \in \mathbb{R}^{2},
$$

y $z$ fija solamente el estado oculto inicial:

$$
h_0 = W_{\mathrm{init}}\, z + b_{\mathrm{init}}, \qquad
h_k = \mathrm{GRU}\big(\varphi(t_k);\, h_{k-1}\big), \qquad
\hat Z(t_k) = W_{\mathrm{out}}\, h_k + b_{\mathrm{out}} .
$$

La pérdida es el error cuadrático medio sobre los elementos del ciclo estandarizado,

$$
\mathcal{L}_{\mathrm{rec}} = \frac{1}{606\,N}\sum_{i=1}^{N} \big\Vert \hat Z_i - Z_i \big\Vert_F^{2} .
$$

## 2. Modificación 1: el código en cada paso del decodificador

Con la entrada constante $\varphi(t_k)$, la información del ciclo solamente llega por el estado inicial y tiende a diluirse a lo largo de 101 pasos. En la versión refinada, $z$ se concatena a la entrada de cada paso:

$$
u_k = \big[\varphi(t_k)\,\Vert\, z\big] \in \mathbb{R}^{2+d}, \qquad
h_k = \mathrm{GRU}\big(u_k;\, h_{k-1}\big), \qquad h_0 = W_{\mathrm{init}}\, z + b_{\mathrm{init}} .
$$

El codificador, el número de unidades ocultas (48) y la capa de salida no cambian.

## 3. Modificación 2: pérdida de cierre del ciclo

Un ciclo de marcha es periódico, de modo que $\hat Z(t_0)$ y $\hat Z(t_{100})$ deben coincidir. Se añade el promedio del salto de cierre en la escala estandarizada:

$$
\mathcal{L}_{\mathrm{cierre}} = \frac{1}{6N}\sum_{i=1}^{N}\sum_{j=1}^{6}
\Big|\hat Z_i^{(j)}(t_0) - \hat Z_i^{(j)}(t_{100})\Big| ,
\qquad
\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + w\,\mathcal{L}_{\mathrm{cierre}} .
$$

Los pesos usados son $w = 0.1$ con $d = 32$ para el modelo A, y $w = 0.05$ con $d = 16$ para el modelo B. La misma pérdida se evalúa sobre validación como criterio de parada temprana. Con $w = 0$ y sin la modificación 1 se recupera el modelo original.

## 4. Modelos A y B

El modelo A se entrena con ciclos limpios, $\hat Z_i = f_\theta(Z_i)$. El modelo B es la misma red entrenada con una versión corrompida de la entrada y con el ciclo limpio como objetivo, $\hat Z_i = f_\theta(\tilde Z_i)$. Para cada articulación $j$,

$$
c_i^{(j)} = \frac{1}{101}\sum_{k} Z_i^{(j)}(t_k), \qquad
a_i^{(j)} \sim \mathcal{U}(0.5,\,1.8), \qquad
o_i^{(j)} \sim \mathcal{U}(-0.9,\,0.9),
$$

$$
\tilde Z_i^{(j)}(t) = c_i^{(j)} + a_i^{(j)}\big(Z_i^{(j)}(t) - c_i^{(j)}\big) + o_i^{(j)} + \varepsilon(t), \qquad \varepsilon(t) \sim \mathcal{N}(0,\,0.15^{2}),
$$

seguido de un desplazamiento circular independiente para el lado izquierdo (canales 1 a 3) y el derecho (canales 4 a 6), con desplazamientos enteros uniformes en $\{-8,\dots,8\}$. En entrenamiento se genera una corrupción nueva en cada lote. En la evaluación II se usa una única realización fija (semilla 42) sobre los ciclos de control de prueba.

## 5. Codificador fundacional congelado

Se sustituye el codificador recurrente por las representaciones internas de un modelo preentrenado de series de tiempo, que permanece congelado. Cada canal $j$ se trata como una serie univariada $s^{(j)} \in \mathbb{R}^{101}$ y se remuestrea linealmente a 128 puntos, que forman $P = 4$ parches de 32 puntos. El modelo normaliza cada serie por instancia con estadísticos acumulados por parche, lo que elimina nivel y escala; por ello se conservan la media $m_j$ y la desviación $v_j$ de la serie. Con $h^{(\ell)}_p \in \mathbb{R}^{1280}$ el estado oculto del bloque $\ell$ en el parche $p$, el vector de características del canal es

$$
\phi_j = \Big[\ \tfrac{1}{P}\textstyle\sum_{p=1}^{P} h^{(\ell)}_p\ \Big\Vert\ h^{(\ell)}_P\ \Big\Vert\ m_j\ \Big\Vert\ v_j\ \Big] \in \mathbb{R}^{2562},
\qquad f = [\phi_1\Vert\cdots\Vert\phi_6] \in \mathbb{R}^{15372}.
$$

Las características se estandarizan con las medias $m_f$ y desviaciones $s_f$ del conjunto de entrenamiento, y una capa lineal entrenable produce el código:

$$
\tilde f = (f - m_f)\oslash s_f, \qquad z = W_{h}\,\tilde f + b_{h} .
$$

El decodificador es el refinado de las secciones 2 y 3, entrenado desde cero con la misma pérdida y el mismo protocolo. En el modelo B, $f$ se recalcula en cada lote sobre el ciclo corrupto.

Se emplean dos controles con el mismo decodificador: pesos aleatorios con la misma arquitectura (separa el aporte del preentrenamiento del de la arquitectura) y la serie estandarizada aplanada, $f = \mathrm{vec}(Z) \in \mathbb{R}^{606}$, sin modelo preentrenado.

## 6. Métricas

Con $\hat y$ y $y$ en grados, $\mathrm{MAE} = \frac{1}{n}\sum|\hat y - y|$ y $\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum(\hat y - y)^2}$. El salto de cierre es

$$
\mathrm{salto} = \frac{1}{N}\sum_{i}\big|\hat G_i(t_0) - \hat G_i(t_{100})\big| ,
$$

promediado también sobre articulaciones. Se añaden la diferencia de pendientes en la costura, $|(\hat G(t_1)-\hat G(t_0)) - (\hat G(t_{100})-\hat G(t_{99}))|$, el error de los incrementos entre pasos consecutivos y la rugosidad, definida como el valor absoluto medio de la segunda diferencia de la predicción.

La ganancia de corrección (evaluación II) y la ganancia de tipicidad (evaluación III) son

$$
\mathrm{ganancia}_{\mathrm{corr}} = 1 - \frac{\mathrm{MAE}(\hat G(\tilde G),\, G)}{\mathrm{MAE}(\tilde G,\, G)}, \qquad
\mathrm{ganancia}_{\mathrm{tip}} = \frac{1}{N}\sum_i \Big(1 - \frac{d(\hat Z_i)}{d(Z_i)}\Big),
$$

con $d(x) = \min_{i \in \mathrm{train}} \Vert x - Z_i \Vert_2$ la distancia al control de entrenamiento más cercano en el espacio estandarizado. La intervención es el error absoluto medio entre la salida y la entrada sobre los casos externos, en grados. Ningún valor de estas métricas debe interpretarse como severidad clínica validada.
