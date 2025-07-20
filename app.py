from dotenv import load_dotenv

load_dotenv()  # Esto carga las variables del archivo .env en el entorno del sistema

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import json  # Importar json para manejar el archivo de metadatos de proyectos
import uuid  # Importar uuid para generar IDs únicos para los proyectos
from urllib.parse import urlparse # Importar urlparse para procesar URLs de GitHub

app = Flask(__name__)
# ¡IMPORTANTE! Cambia esto por una clave secreta fuerte y única.
# Utiliza una variable de entorno en producción.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads', 'projects') # Carpeta donde se guardan los archivos de los proyectos
app.config['ALLOWED_EXTENSIONS'] = {'py', 'ipynb', 'zip', 'pdf', 'txt', 'md', 'rar', 'png', 'jpg', 'jpeg', 'gif'}

# Inicializa Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
# Define la vista de login a la que se redirigirá si no estás autenticado
login_manager.login_view = 'login'


# --- Gestión de Usuarios (Simplificada para un único usuario administrador) ---
class User(UserMixin):
    def __init__(self, id):
        self.id = id

    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    @property
    def is_admin(self):
        """
        Simula un campo is_admin para el usuario administrador codificado.
        Solo el usuario con ADMIN_USERNAME será considerado administrador.
        """
        return self.id == ADMIN_USERNAME # <--- AÑADE ESTA PROPIEDAD




# En una aplicación real, cargarías el usuario desde una base de datos.
# Por simplicidad, aquí usaremos un único usuario admin codificado.
# ¡NUNCA codifiques contraseñas en una aplicación real! Usa contraseñas hasheadas y variables de entorno.
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')


@login_manager.user_loader
def load_user(user_id):
    """
    Callback usado por Flask-Login para recargar el objeto de usuario
    desde el ID de usuario almacenado en la sesión.
    """
    if user_id == ADMIN_USERNAME:
        return User(ADMIN_USERNAME)
    return None


# --- Rutas de la Aplicación ---

# Función para verificar extensiones permitidas
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# --- Configuración y Carga de Proyectos ---
# Lista donde se almacenarán los metadatos de los proyectos
# Se carga desde un archivo JSON o se inicializa con proyectos por defecto.
PROJECTS_FILE = 'projects_metadata.json'  # Archivo para guardar metadatos de proyectos

projects_data = []  # Inicializa la lista vacía

if os.path.exists(PROJECTS_FILE):
    try:
        with open(PROJECTS_FILE, 'r') as f:
            projects_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Advertencia: El archivo '{PROJECTS_FILE}' está corrupto o vacío. Iniciando con proyectos vacíos.")
        projects_data = []  # Si el JSON está corrupto, empieza vacío
else:
    # Aquí puedes precargar tu proyecto de MovieLens si quieres que siempre aparezca
    # y si el archivo de metadatos no existe aún.
    projects_data = [
        {
            'id': 'movielens-recommender',
            'name': 'Sistema de Recomendación de Películas Personalizado (MovieLens 25M)',
            'description': 'Desarrollo e implementación de un sistema de recomendación utilizando Apache Spark (PySpark) y el modelo ALS para procesamiento de Big Data. Incluye despliegue con Docker (API Flask/FastAPI e interfaz web), demostrando experiencia en el ciclo completo de proyectos de IA y procesamiento de datos a gran escala. **Este proyecto demuestra mis habilidades en:** Big Data (Spark), Machine Learning (ALS), Desarrollo de APIs (Flask/FastAPI), Contenerización (Docker), y Despliegue en la nube.',
            'demo_url': 'https://tu-url-de-demo-en-vivo.com',  # ¡AQUÍ VA LA URL PÚBLICA REAL DE TU DEMO!
            'github_url': 'https://github.com/tu_usuario/tu_repo_movielens',  # Reemplaza con tu repo de GitHub
            'report_url': url_for('static', filename='docs/informe_movielens.pdf') if os.path.exists(
                os.path.join(app.root_path, 'static', 'docs', 'informe_movielens.pdf')) else None
        },
        # Puedes añadir más proyectos aquí si quieres que estén por defecto
    ]
    # Guarda estos proyectos iniciales en el archivo JSON si el archivo no existía
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(projects_data, f, indent=4)


# --- Fin Configuración y Carga de Proyectos ---


@app.route('/')
def index():
    """
    Ruta principal del portafolio, visible para todos.
    """
    return render_template('index.html', projects=projects_data)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Ruta para iniciar sesión.
    """
    # Si el usuario ya está autenticado, redirige a la página principal
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            user = User(ADMIN_USERNAME)
            login_user(user)  # Inicia la sesión del usuario
            flash('¡Sesión iniciada correctamente!', 'success')
            return redirect(url_for('upload_project'))  # Redirige a la página de subida
        else:
            flash('Usuario o contraseña inválidos.', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required  # Requiere que el usuario esté autenticado para cerrar sesión
def logout():
    """
    Ruta para cerrar sesión.
    """
    logout_user()  # Cierra la sesión del usuario
    flash('Has cerrado sesión.', 'info')
    return redirect(url_for('index'))

@app.route('/projects') # ¡NUEVA RUTA PARA PROYECTOS!
def list_projects():
    """
    Ruta para mostrar la lista completa de proyectos.
    """
    # Procesar la URL de GitHub para mostrarla de forma limpia
    for project in projects_data:
        if project.get('github_url'):
            try:
                parsed_url = urlparse(project['github_url'])
                # Ej: github.com/usuario/repositorio
                project['github_for_display'] = f"{parsed_url.netloc}{parsed_url.path}"
            except Exception: # Usamos Exception genérica por si urlparse falla por alguna razón
                project['github_for_display'] = project['github_url'] # Si hay error, usa la URL completa
        else:
            project['github_for_display'] = None # No hay URL de GitHub

    # Renderiza una nueva plantilla que solo mostrará los proyectos.
    return render_template('projects.html', projects=projects_data)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_project():
    """
    Ruta para subir proyectos. Solo accesible para administradores.
    """
    if request.method == 'POST':
        # Obtener datos del formulario
        project_name = request.form['project_name']
        description = request.form['description']

        demo_url = request.form.get('demo_url', '')
        github_url = request.form.get('github_url', '')
        report_url = request.form.get('report_url', '')

        # --- Lógica para el ARCHIVO PRINCIPAL del proyecto (ya la tienes) ---
        if 'project_file' not in request.files:
            flash('No se encontró la parte del archivo principal del proyecto.', 'danger')
            return redirect(request.url)
        file = request.files['project_file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo principal del proyecto.', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename): # Asegúrate de que esta función 'allowed_file' esté definida
            flash('Tipo de archivo principal del proyecto no permitido.', 'danger')
            return redirect(request.url)

        # Generar un ID único para el proyecto
        project_id = str(uuid.uuid4())
        original_filename = file.filename
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        stored_filename = f"{project_id}.{file_ext}" # Nombre del archivo principal en el servidor

        # Guardar el archivo principal
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], stored_filename))

        # --- AQUÍ EMPIEZA LA NUEVA LÓGICA PARA LA IMAGEN ---
        project_image_filename = None # Inicializa a None
        if 'project_image' in request.files:
            image_file = request.files['project_image']
            if image_file.filename != '':
                # Verifica la extensión de la imagen usando allowed_file
                if allowed_file(image_file.filename):
                    image_ext = image_file.filename.rsplit('.', 1)[1].lower()
                    project_image_filename = f"{project_id}_img.{image_ext}" # Nombre único para la imagen
                    image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], project_image_filename))
                else:
                    flash('Tipo de archivo de imagen no permitido. Se subirá el proyecto sin imagen.', 'warning')
        # --- AQUÍ TERMINA LA NUEVA LÓGICA PARA LA IMAGEN ---

        # Crear el diccionario del nuevo proyecto
        new_project = {
            'id': project_id,
            'name': project_name,
            'description': description,
            'filename_on_server': stored_filename,
            'original_filename': original_filename,
            'demo_url': demo_url if demo_url else None,
            'github_url': github_url if github_url else None,
            'report_url': report_url if report_url else None,
            'image_filename': project_image_filename # <--- ¡AÑADE ESTA LÍNEA AL DICCIONARIO!
        }

        projects_data.append(new_project)

        # Guardar los metadatos actualizados en el archivo JSON
        with open(PROJECTS_FILE, 'w') as f:
            json.dump(projects_data, f, indent=4)

        flash('¡Proyecto subido correctamente!', 'success')
        return redirect(url_for('index'))
    # else (para GET request)
    return render_template('upload.html')


@app.route('/project/<id>')  # Ahora usa el ID único del proyecto
def show_project_detail(id):
    """
    Ruta para mostrar detalles de un proyecto específico (visible para todos).
    """
    # Busca el proyecto por su ID en la lista projects_data
    project = next((p for p in projects_data if p['id'] == id), None)
    if project is None:
        flash('Proyecto no encontrado.', 'danger')
        return redirect(url_for('index'))

    # Prepara la URL para descargar el archivo del proyecto, si existe y es descargable
    download_project_file_url = None
    # Verifica si el proyecto tiene un archivo asociado y si existe en el servidor
    if 'filename_on_server' in project and project['filename_on_server']:
        file_path_on_disk = os.path.join(app.config['UPLOAD_FOLDER'], project['filename_on_server'])
        if os.path.exists(file_path_on_disk):
            download_project_file_url = url_for('download_project_file', project_id=project['id'])

    return render_template('project_detail.html', project=project, download_project_file_url=download_project_file_url)


@app.route('/download-project-file/<project_id>')
def download_project_file(project_id):
    """
    Ruta para descargar el archivo de un proyecto específico.
    """
    project = next((p for p in projects_data if p['id'] == project_id), None)
    if project and 'filename_on_server' in project:
        # Define el directorio donde se encuentran los archivos de los proyectos
        directory = app.root_path + '/' + app.config['UPLOAD_FOLDER']
        # Envía el archivo para descarga, usando el nombre original del archivo para el usuario
        return send_from_directory(
            directory=directory,
            path=project['filename_on_server'],
            as_attachment=True,
            download_name=project['original_filename']  # Nombre con el que se descarga el archivo
        )
    flash('Archivo de proyecto no encontrado o no disponible.', 'danger')
    return redirect(url_for('index'))


@app.route('/delete-project/<id>', methods=['POST'])
@login_required
def delete_project(id):
    """
    Ruta para eliminar un proyecto y su archivo asociado.
    """
    if not current_user.is_admin:
        flash('No tienes permiso para realizar esta acción.', 'error')
        return redirect(url_for('list_projects'))

    global projects_data

    project_found = None
    for i, project in enumerate(projects_data):
        if project['id'] == id:
            project_found = project
            projects_data.pop(i)
            break

    if project_found:
        # Eliminar el archivo principal del servidor (ya lo tienes)
        if 'filename_on_server' in project_found and project_found['filename_on_server']:
            file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], project_found['filename_on_server'])
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    flash(f'Archivo "{project_found["original_filename"]}" eliminado del servidor.', 'info')
                except OSError as e:
                    flash(f'Error al eliminar el archivo principal: {e}', 'danger')
                    print(f"Error al eliminar archivo {file_path}: {e}")
            else:
                flash(f'Advertencia: Archivo principal "{project_found["original_filename"]}" no encontrado en la carpeta de subida.', 'warning')

        # --- AÑADE ESTA NUEVA LÓGICA PARA ELIMINAR LA IMAGEN ASOCIADA ---
        if 'image_filename' in project_found and project_found['image_filename']:
            image_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], project_found['image_filename'])
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    flash(f'Imagen asociada "{project_found["image_filename"]}" eliminada del servidor.', 'info')
                except OSError as e:
                    flash(f'Error al eliminar la imagen asociada: {e}', 'danger')
                    print(f"Error al eliminar imagen {image_path}: {e}")
            else:
                flash(f'Advertencia: Imagen asociada "{project_found["image_filename"]}" no encontrada en la carpeta de subida.', 'warning')
        # ----------------------------------------------------------------

        # Guardar los metadatos actualizados en el archivo JSON
        try:
            with open(PROJECTS_FILE, 'w') as f:
                json.dump(projects_data, f, indent=4)
            flash('Proyecto eliminado correctamente.', 'success')
        except Exception as e:
            flash(f'Error al guardar los metadatos del proyecto: {e}', 'danger')
            print(f"Error al guardar proyectos_data en JSON: {e}")
            projects_data.append(project_found) # Esto sería una lógica de rollback
    else:
        flash('Proyecto no encontrado para eliminar.', 'danger')

    return redirect(url_for('list_projects')) # Redirige a la lista de proyectos después de la eliminación (tu projects.html)

# --- RUTAS PARA EL CV ---
@app.route('/cv')
def show_cv():
    """
    Ruta para mostrar la página del CV en formato HTML.
    """
    return render_template('cv.html')


@app.route('/download-cv')
def download_cv():
    """
    Ruta para descargar el archivo CV directamente.
    """
    directory = app.root_path + '/static/docs/'
    filename = 'Francisco_Calero_CV.pdf'
    return send_from_directory(directory=directory, path=filename, as_attachment=True)


# --- FIN RUTAS PARA EL CV ---


# --- Bloque principal de ejecución ---
if __name__ == '__main__':
    # Asegúrate de que las carpetas necesarias existan al inicio de la aplicación
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Carpeta para proyectos subidos
    os.makedirs(os.path.join(app.root_path, 'static', 'docs'),
                exist_ok=True)  # Carpeta para documentos (PDF del CV, informes)
    os.makedirs(os.path.join(app.root_path, 'static', 'img'),
                exist_ok=True)  # Carpeta para imágenes (JPG del CV, otras imágenes)
    app.run(debug=True)  # En producción, desactiva debug=True para mayor seguridad y rendimiento.