# ===============================================================================
# --- Importaciones Necesarias ---
# Se importan las clases y funciones de PySpark necesarias para manipulación
# de datos (DataFrame), Machine Learning (ALS), evaluación (RegressionEvaluator)
# y funciones SQL (col, avg, count, lit, explode).
# 'os' y 'sys' se usan para la gestión de rutas y depuración del sistema.
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import col, avg, count, lit, explode
import os
import sys

# ===============================================================================
# --- 1. CONFIGURACIÓN DE RUTAS ---
# Se definen las rutas base del proyecto y las ubicaciones específicas
# de los archivos de datos (ratings.csv, movies.csv) y la ruta donde se
# guardará/cargará el modelo ALS entrenado.
# BASE_DIR: Ruta absoluta del directorio donde se encuentra este script.
# DATA_PATH: Subdirectorio 'ml-25m' dentro del directorio base.
# RATINGS_PATH, MOVIES_PATH: Rutas completas a los archivos CSV.
# MODEL_SAVE_LOAD_PATH: Ruta completa para guardar/cargar el modelo. Se crea
#                       un subdirectorio 'models' si no existe.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ml-25m")

RATINGS_PATH = os.path.join(DATA_PATH, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_PATH, "movies.csv")

MODEL_SAVE_LOAD_PATH = os.path.join(BASE_DIR, "models", "als_movie_recommender_model")

# --- Debugging: Verificar Variables de Entorno (Opcional, para información) ---
# Estas líneas imprimen el estado de algunas variables de entorno relevantes
# dentro del contenedor Docker. Son útiles para la depuración y para entender
# cómo Spark percibe el entorno.
print("\n--- Verificando Variables de Entorno (en Docker) ---")
hadoop_home_env = os.environ.get('HADOOP_HOME')
if hadoop_home_env:
    print(f"HADOOP_HOME: {hadoop_home_env}")
else:
    print("HADOOP_HOME: No definida (esperado en este setup Docker).")

python_path_env = os.environ.get('PATH')
if python_path_env:
    print(f"PATH (primeros 200 chars): {python_path_env[:200]}...")
else:
    print("PATH: No definida en este entorno.")

java_home_env = os.environ.get('JAVA_HOME')
if java_home_env:
    print(f"JAVA_HOME: {java_home_env}")
else:
    print("JAVA_HOME: No definida (esperado que Spark la maneje internamente).")
print("--- Fin Verificación Variables de Entorno ---\n")

# --- Asegurarse de que la carpeta base para los modelos exista ---
# Se verifica si el directorio donde se guardarán los modelos existe.
# Si no, se crea. Esto previene errores de escritura al intentar guardar el modelo.
model_base_dir = os.path.dirname(MODEL_SAVE_LOAD_PATH)
if not os.path.exists(model_base_dir):
    print(f"Creando directorio para modelos: {model_base_dir}")
    os.makedirs(model_base_dir, exist_ok=True)
else:
    print(f"Directorio de modelos ya existe: {model_base_dir}")

# ===============================================================================
# --- 2. INICIALIZAR SPARK SESSION ---
# Se inicializa la sesión de Spark, que es el punto de entrada para usar la API de Spark.
# .appName: Nombre de la aplicación Spark.
# .master("local[*]"): Ejecuta Spark en modo local, usando todos los núcleos disponibles.
# .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties"):
#   Configura las opciones de Java para el driver de Spark, incluyendo la configuración
#   de logging desde un archivo 'log4j.properties'.
# .config("spark.driver.memory", "24g"): Asigna 24 GB de memoria al driver de Spark.
# .config("spark.executor.memory", "24g"): Asigna 24 GB de memoria a los ejecutores de Spark.
#   Estas configuraciones de memoria son CRÍTICAS para manejar grandes datasets.
print("Inicializando Spark Session...")
spark = SparkSession.builder \
    .appName("MovieRecommendationSystem") \
    .master("local[*]") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties") \
    .config("spark.driver.memory", "24g") \
    .config("spark.executor.memory", "24g") \
    .getOrCreate()
print("Spark Session inicializada.")

# ===============================================================================
# --- 3. CONFIGURAR NIVEL DE LOGGING DE SPARK ---
# Se establece el nivel de logging de Spark a "WARN" (Advertencia).
# Esto reduce la verbosidad de los logs, mostrando solo advertencias y errores,
# a menos que se cambie a "INFO" para depuración detallada.
spark.sparkContext.setLogLevel("WARN")  # Cambiar a "INFO" para más detalles en depuración

# ===============================================================================
# --- 4. Cargar datos de películas (necesario para recomendaciones) ---
# Se carga el archivo 'movies.csv' en un DataFrame de Spark.
# header=True: Indica que la primera fila contiene los nombres de las columnas.
# inferSchema=True: Spark intentará deducir automáticamente los tipos de datos de las columnas.
# Se seleccionan solo las columnas 'movieId', 'title' y 'genres' para el DataFrame final.
print(f"Cargando movies.csv desde: {MOVIES_PATH}")
movies_df = spark.read.csv(MOVIES_PATH, header=True, inferSchema=True)
movies_selected_df = movies_df.select(col("movieId"), col("title"), col("genres"))
print("movies.csv cargado exitosamente.")

# ===============================================================================
# --- 5. COMPROBAR Y CARGAR/ENTRENAR EL MODELO ---
# Esta es una sección clave que decide si se entrena un nuevo modelo o se carga uno existente.
# model_trained_now: Indicador booleano que se usa más adelante para decidir si evaluar
#                    el modelo (solo si se entrenó en esta ejecución).
model = None
model_trained_now = False

print("\nVerificando si el modelo ALS ya está entrenado...")
try:
    # Intenta cargar el modelo ALS desde la ruta predefinida.
    model = ALSModel.load(MODEL_SAVE_LOAD_PATH)
    print(f"Modelo ALS cargado exitosamente desde: {MODEL_SAVE_LOAD_PATH}")
    model_trained_now = False  # Si se carga, no se entrenó ahora.
except Exception as e:
    # Si la carga falla (ej. el archivo no existe, o hay un error de formato),
    # se procede al entrenamiento.
    print(f"No se encontró un modelo entrenado o hubo un error al cargar: {e}")
    print("Iniciando el proceso de entrenamiento del modelo ALS...")
    model_trained_now = True  # Se entrenará un nuevo modelo.

    # --- Cargar y preprocesar datos de ratings (SOLO SI SE VA A ENTRENAR) ---
    # Los datos de ratings se cargan solo cuando el modelo necesita ser entrenado.
    print(f"Cargando ratings.csv desde: {RATINGS_PATH}")
    ratings_df = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True)
    print("ratings.csv cargado exitosamente.")

    # --- Muestreo del Dataset (CRUCIAL para grandes volúmenes de datos) ---
    # Para manejar el dataset MovieLens 25M en un entorno local, se realiza
    # un muestreo (sampling) del 5% de los datos. Esto reduce drásticamente
    # el uso de memoria durante el entrenamiento, evitando OutOfMemoryError.
    # 'withReplacement=False': Muestreo sin reemplazo.
    # 'fraction=0.05': Selecciona el 5% de las filas.
    # 'seed=42': Asegura que el muestreo sea reproducible.
    sampled_ratings_df = ratings_df.sample(withReplacement=False, fraction=0.05, seed=42)
    print(f"Dataset de ratings muestreado. Nuevo tamaño: {sampled_ratings_df.count()} registros.")

    # Se seleccionan y renombran las columnas necesarias del DataFrame muestreado.
    # El modelo ALS requiere columnas de 'userId', 'movieId' y 'rating'.
    ratings_cleaned_df = sampled_ratings_df.select(  # <-- Se usa el DataFrame muestreado aquí
        col("userId"),
        col("movieId"),
        col("rating").alias("user_rating"),
        col("timestamp")
    )
    print("Ratings preprocesados.")

    # --- Preparación para ALS ---
    # Se crea un DataFrame 'als_data' con las columnas requeridas por el algoritmo ALS.
    als_data = ratings_cleaned_df.select(
        col("userId"),
        col("movieId"),
        col("user_rating").alias("rating")
    )

    # Dividir los datos en conjuntos de entrenamiento y prueba.
    # Se utiliza un split 80/20 para entrenar y evaluar el modelo.
    (training, test) = als_data.randomSplit([0.8, 0.2], seed=42)

    print(f"Tamaño del conjunto de entrenamiento: {training.count()}")
    print(f"Tamaño del conjunto de prueba: {test.count()}")

    # --- Construcción y entrenamiento del modelo ALS ---
    # Se inicializa el objeto ALS (Alternating Least Squares) con sus parámetros:
    # rank: El número de factores latentes en el modelo.
    # maxIter: El número máximo de iteraciones para el algoritmo.
    # regParam: Parámetro de regularización para evitar overfitting.
    # userCol, itemCol, ratingCol: Nombres de las columnas de usuario, ítem y rating.
    # coldStartStrategy="drop": Ignora las predicciones para nuevos usuarios/ítems
    #                           que no estaban en los datos de entrenamiento.
    # seed: Semilla para reproducibilidad.
    als = ALS(
        rank=10,
        maxIter=10,
        regParam=0.01,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        seed=42
    )
    # Se entrena el modelo ALS usando el conjunto de entrenamiento.
    model = als.fit(training)
    print("Modelo ALS entrenado exitosamente.")

    # --- Guardar el modelo entrenado ---
    # Una vez entrenado, el modelo se guarda en el disco para su uso futuro,
    # evitando la necesidad de re-entrenar cada vez.
    # .overwrite(): Permite sobrescribir un modelo existente en la misma ruta.
    print(f"Guardando el modelo ALS en: {MODEL_SAVE_LOAD_PATH}")
    try:
        model.write().overwrite().save(MODEL_SAVE_LOAD_PATH)
        print("Modelo guardado exitosamente.")
    except Exception as e_save:
        # Manejo de errores en caso de fallo al guardar el modelo.
        print(f"ERROR: Fallo al guardar el modelo. Causa: {e_save}")
        print("Esto podría indicar un problema persistente con permisos de escritura.")

# ===============================================================================
# --- 6. EVALUAR EL MODELO (SOLO SI SE ENTRENÓ AHORA) ---
# Si el modelo se entrenó en la ejecución actual (model_trained_now es True),
# se procede a evaluarlo usando el conjunto de prueba.
# Si el modelo fue cargado, se asume que ya fue evaluado en su momento.
if model_trained_now:
    print("\nEvaluando el modelo ALS (recientemente entrenado)...")
    # Se recarga el DataFrame completo de ratings para asegurar que el conjunto de prueba
    # se genere consistentemente para la evaluación (esto es una precaución).
    full_ratings_df = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True)
    # Se aplica el mismo muestreo para el conjunto de evaluación, para que coincida con el entrenamiento.
    sampled_full_ratings_df = full_ratings_df.sample(withReplacement=False, fraction=0.05, seed=42)
    full_ratings_cleaned_df = sampled_full_ratings_df.select(
        col("userId"),
        col("movieId"),
        col("rating").alias("user_rating")
    )
    full_als_data = full_ratings_cleaned_df.select(
        col("userId"),
        col("movieId"),
        col("user_rating").alias("rating")
    )
    # Se divide el dataset completo (muestreado) nuevamente para obtener el conjunto de prueba.
    _, test = full_als_data.randomSplit([0.8, 0.2], seed=42)

    # Se generan predicciones sobre el conjunto de prueba.
    predictions = model.transform(test)
    # Se utiliza RegressionEvaluator para calcular el RMSE (Root Mean Squared Error).
    # El RMSE mide la diferencia entre los valores predichos y los reales.
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Raíz del error cuadrático medio (RMSE) del modelo ALS: {rmse}")
else:
    print("\nEl modelo fue cargado, no re-evaluado en esta ejecución.")

# ===============================================================================
# --- 7. GENERAR RECOMENDACIONES ---
# Sección dedicada a generar las top N recomendaciones para un usuario específico.
print("\nGenerando recomendaciones para usuarios específicos...")

# Se define el ID del usuario para el cual se generarán las recomendaciones.
user_id_to_recommend = 1

# Se carga ratings_df si no se cargó en el bloque de entrenamiento (es decir,
# si el modelo fue cargado y no entrenado en esta ejecución).
try:
    _ = ratings_df  # Intenta acceder a ratings_df
except NameError:
    print("Cargando ratings.csv para determinar películas vistas (necesario para filtrar recomendaciones)...")
    ratings_df = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True)
    # Si se acaba de cargar ratings_df aquí, se debe muestrear para consistencia.
    sampled_ratings_df = ratings_df.sample(withReplacement=False, fraction=0.05, seed=42)
    ratings_df = sampled_ratings_df  # Usar la versión muestreada para el filtro

# 1. Identificar las películas que el usuario ya ha visto.
movies_seen_by_user = ratings_df.filter(col("userId") == user_id_to_recommend) \
    .select("movieId") \
    .distinct()

# 2. Encontrar todas las películas que el usuario *no* ha visto.
all_movie_ids = movies_selected_df.select("movieId").distinct()
movies_not_seen_by_user = all_movie_ids.join(
    movies_seen_by_user,
    on="movieId",
    how="left_anti"
    # left_anti-join devuelve filas de 'all_movie_ids' que no tienen coincidencia en 'movies_seen_by_user'
)

# 3. Crear un DataFrame de prueba para el usuario con las películas no vistas.
# Se añade una columna con el ID del usuario a todas las películas no vistas.
user_unrated_movies = movies_not_seen_by_user.withColumn("userId", lit(user_id_to_recommend))

# 4. Generar predicciones para el usuario en las películas no vistas.
# El modelo ALS predice las calificaciones que el usuario daría a estas películas.
user_predictions = model.transform(user_unrated_movies)

# 5. Unir las predicciones con los títulos de las películas y seleccionar las 10 mejores.
# Se unen las predicciones con el DataFrame de películas para obtener los títulos y géneros.
# Se ordenan las predicciones de forma descendente y se seleccionan las 10 mejores.
top_10_recs_user = user_predictions.join(
    movies_selected_df,
    on="movieId",
    how="inner"
).orderBy(
    col("prediction").desc()
).limit(10)

print(f"\nTop 10 recomendaciones para el usuario {user_id_to_recommend}:")
top_10_recs_user.show(truncate=False)

# ===============================================================================
# --- 8. DETENER SPARK SESSION ---
# Es crucial detener la sesión de Spark al finalizar para liberar los recursos.
print("\nDeteniendo Spark Session...")
spark.stop()
print("Spark Session detenida. Script finalizado.")