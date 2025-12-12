# Análisis de polarización ideológica en Twitter
**Proyecto Final - Tratamiento de Datos**

Patricia Barbero Rodríguez 100363955
Gorka Bernad Santos 100451457
Macarena Fernández Rodríguez 100384038
Belén Larrabeiti Martínez 10384072

---

## Descripción del problema

Este proyecto estudia la relación entre contenido ideológico y patrones lingüísticos en twitter con el objetivo de:

- Identificar la orientación ideológica de distintas publicaciones.
- Analizar la polarización ideológica mediante los embeddings.
- Comparar estrategias de modelado y embeddings.

Para ello utilizamos el conjunto de datos: POLITiCES 2023: Political Ideology and Power in Spanish Society (dado en el documento)

**Formado por:**

- Número de filas: 14400
- Número de columnas: 6

**Tipos de datos:**

label                  object
gender                 object
profession             object
ideology_binary        object
ideology_multiclass    object
tweet                  object
dtype: object

Valores nulos por columna:
label                  0
gender                 0
profession             0
ideology_binary        0
ideology_multiclass    0
tweet                  0
dtype: int64

Valores únicos por columna:
label                    360
gender                     2
profession                 3
ideology_binary            2
ideology_multiclass        4
tweet                  14384
dtype: int64

[ ]

**OBSERVACIONES:**

- Dataset desbalanceado (clases left y right están subrepresentadas)
- Datos muy genéricos (tweets que no muestran ideología en su mayoría)

---

## Metodología

### Parte 1: Análisis exploratorio del conjunto de datos

Se ha realizado un análisis completo de los datos que incluye:

- Distribución del número de tweets por variable mediante gráficas de barras y gráfica quesito.
- Distribución de ideologías mediante barras subdivididas por género, por profesión, y un histograma con la distribución de longitud de tweets.
- Nubes de texto antes de la limpieza de vocabulario para poder visualizar las diferencias.

### Parte 2: Representación vectorial del texto

Primero se ha realizado un preprocesado del texto, es decir, se ha aplicado una función de limpieza aplicada a todos los tweets para quitar palabras como preposiciones, @user, urls etc que pueden obstaculizar el entrenamiento y análisis.

Se han comparado tres estrategias de representación diferentes:

#### 1) TF-IDF

Se utilizó un modelo TF-IDF como representación léxica, implementado con TfidfVectorizer de scikit-learn. Esta representación genera una matriz (sparse), que posteriormente se utilizará para el entrenamiento. 

**Dimensiones resultantes:**

Dimensiones de la matriz TF-IDF train set: (10080, 5000)
Dimensiones de la matriz TF-IDF validation set: (4320, 5000)
Dimensiones de la matriz TF-IDF test set: (3600, 5000)

**Top de palabras:**

gracias 120.42329743534857
gobierno 114.02342568299336
españa 108.04524812075806
años 103.31815510207267
ley 73.8876495764722
madrid 73.55735486730096
gente 62.00059070603916
mundo 57.86676803313677
año 57.616455598756076
país 56.73649979797708
vida 51.558217904449194
política 51.050209914265466
cosas 50.534957525232606
trabajo 46.406209030332995
mujeres 45.45618114939983
presidente 45.32934029053684
izquierda 43.91606330164386
personas 43.69479463570944
partido 43.17438344134408
historia 42.03383842885117

#### 2) Word2Vec

Se entrena con un corpus de tweets limpios, se concatenan el promedio y máximo de embeddings de las palabras. Dado que Word2Vec solo produce embeddings a nivel de palabra, la representación final de cada tweet se ha obtenido mediante el promedio más el máximo, es decir, cada tweet se transformó en un vector concatenando el promedio de los embeddings de todas las palabras y el máximo calculado elemento a elemento para todas las palabras.

#### 3) BERT Embeddings (Contextuales)

Utilizando "sentence-transformers/distiluse-base-multilingual-cased-v2", se obtuvieron embeddings que capturan las relaciones semánticas, desambiguación por contexto y similitud semántica entre frases completas.


## Modelado y evaluación

Se han entrenado, haciendo uso de cada una de las representaciones vectoriales, un regresor logístico, un svm, y una red neuronal para poder evaluar su rendimiento:

### TF-IDF

==== VAL LR + TFIDF ====
                precision    recall  f1-score   support

          left       0.32      0.40      0.36       828
 moderate_left       0.51      0.44      0.47      1572
moderate_right       0.49      0.41      0.45      1416
         right       0.23      0.35      0.28       504

      accuracy                           0.41      4320
     macro avg       0.39      0.40      0.39      4320
  weighted avg       0.44      0.41      0.42      4320

==== TEST LR + TFIDF ====
                precision    recall  f1-score   support

          left       0.27      0.35      0.31       640
 moderate_left       0.49      0.39      0.43      1440
moderate_right       0.43      0.41      0.42      1080
         right       0.23      0.31      0.26       440

      accuracy                           0.38      3600
     macro avg       0.36      0.37      0.36      3600
  weighted avg       0.40      0.38      0.39      3600

==== VAL SVM + TFIDF ====
                precision    recall  f1-score   support

          left       0.31      0.40      0.35       828
 moderate_left       0.51      0.45      0.48      1572
moderate_right       0.50      0.40      0.44      1416
         right       0.24      0.34      0.28       504

      accuracy                           0.41      4320
     macro avg       0.39      0.40      0.39      4320
  weighted avg       0.44      0.41      0.42      4320

==== TEST SVM + TFIDF ====
                precision    recall  f1-score   support

          left       0.26      0.37      0.30       640
 moderate_left       0.48      0.37      0.42      1440
moderate_right       0.42      0.39      0.40      1080
         right       0.22      0.28      0.24       440

      accuracy                           0.36      3600
     macro avg       0.34      0.35      0.34      3600
  weighted avg       0.39      0.36      0.37      3600

=== TEST MLP + TF-IDF ===
             precision    recall  f1-score   support

           0       0.27      0.25      0.26       640
           1       0.47      0.45      0.46      1440
           2       0.39      0.46      0.42      1080
           3       0.25      0.22      0.24       440

    accuracy                           0.39      3600
   macro avg       0.35      0.34      0.34      3600
weighted avg       0.39      0.39      0.39      3600


### Word2Vec

==== VAL LR + Word2Vec ====
                precision    recall  f1-score   support

          left       0.25      0.32      0.29       828
 moderate_left       0.47      0.34      0.40      1572
moderate_right       0.42      0.31      0.36      1416
         right       0.19      0.40      0.26       504

      accuracy                           0.34      4320
     macro avg       0.33      0.34      0.32      4320
  weighted avg       0.38      0.34      0.35      4320

==== TEST LR + Word2Vec ====
                precision    recall  f1-score   support

          left       0.21      0.29      0.24       640
 moderate_left       0.46      0.30      0.36      1440
moderate_right       0.39      0.31      0.35      1080
         right       0.18      0.36      0.24       440

      accuracy                           0.31      3600
     macro avg       0.31      0.32      0.30      3600
  weighted avg       0.36      0.31      0.32      3600
==== VAL SVM + Word2Vec ====
                precision    recall  f1-score   support

          left       0.25      0.36      0.30       828
 moderate_left       0.48      0.34      0.39      1572
moderate_right       0.41      0.28      0.33      1416
         right       0.18      0.38      0.25       504

      accuracy                           0.33      4320
     macro avg       0.33      0.34      0.32      4320
  weighted avg       0.38      0.33      0.34      4320

==== TEST SVM + Word2Vec ====
                precision    recall  f1-score   support

          left       0.21      0.34      0.26       640
 moderate_left       0.48      0.30      0.37      1440
moderate_right       0.40      0.28      0.33      1080
         right       0.19      0.39      0.26       440

      accuracy                           0.31      3600
     macro avg       0.32      0.33      0.30      3600
  weighted avg       0.37      0.31      0.32      3600
=== TEST MLP + Word2Vec ===
              precision    recall  f1-score   support

           0       0.22      0.10      0.14       640
           1       0.46      0.44      0.45      1440
           2       0.43      0.02      0.04      1080
           3       0.17      0.74      0.28       440

    accuracy                           0.29      3600
   macro avg       0.32      0.33      0.23      3600
weighted avg       0.37      0.29      0.25      3600

### BERT

==== VAL LR + BERT ====
                precision    recall  f1-score   support

          left       0.31      0.38      0.34       828
 moderate_left       0.51      0.39      0.44      1572
moderate_right       0.45      0.31      0.37      1416
         right       0.23      0.51      0.32       504

      accuracy                           0.38      4320
     macro avg       0.37      0.40      0.37      4320
  weighted avg       0.42      0.38      0.38      4320

==== TEST LR + BERT ====
                precision    recall  f1-score   support

          left       0.26      0.35      0.30       640
 moderate_left       0.55      0.36      0.43      1440
moderate_right       0.46      0.38      0.41      1080
         right       0.24      0.49      0.32       440

      accuracy                           0.38      3600
     macro avg       0.38      0.39      0.37      3600
  weighted avg       0.43      0.38      0.39      3600
==== VAL SVM + BERT ====
                precision    recall  f1-score   support

          left       0.31      0.39      0.34       828
 moderate_left       0.52      0.37      0.44      1572
moderate_right       0.47      0.32      0.38      1416
         right       0.23      0.55      0.32       504

      accuracy                           0.38      4320
     macro avg       0.38      0.41      0.37      4320
  weighted avg       0.43      0.38      0.39      4320

==== TEST SVM + BERT ====
                precision    recall  f1-score   support

          left       0.27      0.38      0.31       640
 moderate_left       0.56      0.34      0.43      1440
moderate_right       0.46      0.36      0.40      1080
         right       0.23      0.50      0.32       440

      accuracy                           0.37      3600
     macro avg       0.38      0.40      0.36      3600
  weighted avg       0.44      0.37      0.38      3600
=== TEST MLP + BERT ===
              precision    recall  f1-score   support

           0       0.27      0.29      0.28       640
           1       0.52      0.44      0.48      1440
           2       0.39      0.55      0.46      1080
           3       0.30      0.12      0.18       440

    accuracy                           0.41      3600
   macro avg       0.37      0.35      0.35      3600
weighted avg       0.41      0.41      0.40      3600

### Modelo Hugging Face

Resultados en Validation:
  • Accuracy:    0.4567
  • F1-macro:    0.4040
  • F1-weighted: 0.4484
  • Loss:        1.2582
============================================================
CLASSIFICATION REPORT - VALIDATION
============================================================
                precision    recall  f1-score   support

          left       0.33      0.33      0.33       828
 moderate_left       0.51      0.62      0.56      1572
moderate_right       0.49      0.44      0.46      1416
         right       0.34      0.22      0.27       504

      accuracy                           0.46      4320
     macro avg       0.42      0.40      0.40      4320
  weighted avg       0.45      0.46      0.45      4320

============================================================
EVALUACIÓN FINAL CON TEST SET
============================================================

Resultados en Test:
  • Accuracy:    0.4350
  • F1-macro:    0.3788
  • F1-weighted: 0.4296


============================================================
CLASSIFICATION REPORT - TEST
============================================================
                precision    recall  f1-score   support

          left       0.28      0.31      0.29       640
 moderate_left       0.51      0.56      0.54      1440
moderate_right       0.45      0.44      0.45      1080
         right       0.33      0.19      0.24       440

      accuracy                           0.43      3600
     macro avg       0.39      0.38      0.38      3600
  weighted avg       0.43      0.43      0.43      3600

Comparativa final:



Modelo
Accuracy
F1-macro
ROC-AUC
0
LR + TF-IDF
0.379444
0.356050
0.635392
1
SVM + TF-IDF
0.363611
0.341354
0.626499
2
MLP + TF-IDF
0.388611
0.344749
NaN
3
LR + Word2Vec
0.310278
0.298363
0.589339
4
SVM + Word2Vec
0.312778
0.303818
0.591209
5
MLP + Word2Vec
0.290833
0.228021
NaN
6
LR + BERT emb
0.379167
0.367103
0.645298
7
SVM + BERT emb
0.373611
0.363960
0.643901
8
MLP + BERT emb
0.408889
0.348186
NaN


## Parte final: extensión

La extensión escogida para ampliar el proyecto ha sido un análisis más profundo de los patrones lingüísticos que puedan estar asociados a la polarización ideológica en tweets.

1) **Marcadores de polarización**

El objetivo es identificar los elementos lingüísticos específicos distintivos de las ideologías.

Se calcula la diferencia entre el TF-IDF promedio de una ideología vs. las demás.  
Se identifican palabras "divisoras" (términos que aparecen mucho en una ideología y poco en las demás) y "puente" (que se usan por igual en todas las ideologías).

============================================================
Calculando palabras distintivas de: LEFT
============================================================

Top 10 palabras más distintivas:
   1. derecha                   | Score: 0.0044
   2. derechos                  | Score: 0.0043
   3. facua                     | Score: 0.0040
   4. extrema derecha           | Score: 0.0037
   5. fascistas                 | Score: 0.0032
   6. extrema                   | Score: 0.0031
   7. ultraderecha              | Score: 0.0027
   8. violencia                 | Score: 0.0026
   9. cosa                      | Score: 0.0026
  10. barrio                    | Score: 0.0025

============================================================
Calculando palabras distintivas de: MODERATE_LEFT
============================================================

Top 10 palabras más distintivas:
   1. asturias                  | Score: 0.0041
   2. compromiso                | Score: 0.0032
   3. informa                   | Score: 0.0031
   4. gracias                   | Score: 0.0030
   5. mujeres                   | Score: 0.0030
   6. semana                    | Score: 0.0027
   7. importante                | Score: 0.0027
   8. vida                      | Score: 0.0026
   9. reunión                   | Score: 0.0025
  10. año                       | Score: 0.0024

============================================================
Calculando palabras distintivas de: MODERATE_RIGHT
============================================================

Top 10 palabras más distintivas:
   1. gobierno                  | Score: 0.0088
   2. españa                    | Score: 0.0045
   3. impuestos                 | Score: 0.0038
   4. españoles                 | Score: 0.0036
   5. recuadro                  | Score: 0.0035
   6. vd                        | Score: 0.0025
   7. programa                  | Score: 0.0025
   8. ley                       | Score: 0.0025
   9. inflación                 | Score: 0.0025
  10. ministra                  | Score: 0.0024

============================================================
Calculando palabras distintivas de: RIGHT
============================================================

Top 10 palabras más distintivas:
   1. feliz                     | Score: 0.0089
   2. gt                        | Score: 0.0065
   3. beso                      | Score: 0.0048
   4. feliz cumpleaños          | Score: 0.0045
   5. hombre                    | Score: 0.0044
   6. novia                     | Score: 0.0042
   7. feliz beso                | Score: 0.0040
   8. hijos                     | Score: 0.0040
   9. cumpleaños                | Score: 0.0040
  10. pases                     | Score: 0.0038

================================================================================
PALABRAS DIVISORAS (uso muy diferente entre ideologías)
================================================================================

Palabra                     Varianza       Left      Mod-L      Mod-R      Right
--------------------------------------------------------------------------------
gobierno                    0.000312      0.046      0.052      0.089      0.048
españa                      0.000227      0.029      0.052      0.070      0.060
derecha                     0.000132      0.040      0.020      0.014      0.011
feliz                       0.000072      0.003      0.009      0.009      0.026
presidente                  0.000067      0.009      0.027      0.027      0.013
derechos                    0.000061      0.025      0.012      0.008      0.004
ve                          0.000060      0.305      0.288      0.296      0.285
izquierda                   0.000058      0.025      0.007      0.024      0.026
cosa                        0.000054      0.035      0.021      0.024      0.038
gt                          0.000054      0.004      0.006      0.002      0.020
país                        0.000044      0.027      0.038      0.028      0.020
mujer                       0.000040      0.020      0.031      0.017      0.031
españoles                   0.000040      0.005      0.007      0.020      0.015
empresa                     0.000036      0.023      0.020      0.013      0.008
llama                       0.000036      0.024      0.014      0.015      0.028

================================================================================
PALABRAS PUENTE (uso similar en todas las ideologías)
================================================================================

Palabra                     Varianza       Left      Mod-L      Mod-R      Right
--------------------------------------------------------------------------------
problema                    0.000006      0.019      0.013      0.014      0.014


2) **Polarization score**

Para cuantificar el "nivel de polarización" de los tweets individuales se han analizado los embeddings contextuales obtenidos con BERT. Para cada tweet, se ha calculado:

- La distancia al centro mediante la distancia coseno al centro político global (cuanto más alta sea, más extremo o polarizado será)
- La "ideology score", que es la posición respecto a ese centro que hemos calculado como `(dist_left - dist_right) / (dist_left + dist_right)`

============================================================
EJEMPLOS DE POLARIZATION SCORES
============================================================

  Texto: titular titulamos por el asesino nunca por la mujer que ha sido asesinada el foc...
  Ideología real: moderate_left
  Distancia al centro: 0.7124
  Ideology score: -0.0009 (LEFT)
  ¿Es extremo?: NO

 Texto: tengo el deber de q cada uno de los ciudadanos del país tome una decisión libre ...
  Ideología real: moderate_left
  Distancia al centro: 0.6884
  Ideology score: +0.0024 (RIGHT)
  ¿Es extremo?: NO

3) **Análisis de Confusión del Modelo**

Se han identificado los tweets donde el modelo falla.

=======================================================================
TWEETS INCORRECTAMENTE CLASIFICADOS: 2692 de 4320
=======================================================================
Confusiones más frecuentes:
real            predicted     
moderate_right  right             351
                moderate_left     328
moderate_left   moderate_right    325
                left              320
                right             314
moderate_right  left              294
left            right             193
                moderate_left     185
                moderate_right    134
right           moderate_right     89
dtype: int64


4) **Sistema interactivo**

Permite un análisis de tweets que escriba el usuario. Al escribir un tweet se obtiene la predicción ideológica, la polarization score, su posicionamiento relativo en el espectro de left y right, y las palabras clave que se hayan detectado como divisoras, si las hay.

**Ejemplo:**

Escribe un tweet: viva el orden y la ley

TEXTO ORIGINAL:
   viva el orden y la ley

──────────────────────────────────────────────────────────────────────-
PREDICCIÓN IDEOLÓGICA
───────────────────────────────────────────────────────────────────────
   - Ideología predicha: RIGHT

   Distribución de probabilidades:
     left                 [███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 29.29%
     moderate_left        [███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  9.18%
     moderate_right       [█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 23.98%
     right                [███████████████░░░░░░░░░░░░░░░░░░░░░░░░░] 37.55%

───────────────────────────────────────────────────────────────────────
POLARIZATION SCORE
───────────────────────────────────────────────────────────────────────
   - Distancia al centro político: 0.8589

   Espectro ideológico:
     Score: +0.0074

───────────────────────────────────────────────────────────────────────
 PALABRAS CLAVE DETECTADAS
───────────────────────────────────────────────────────────────────────
    No se encontraron palabras divisoras significativas

# Conclusiones finales
