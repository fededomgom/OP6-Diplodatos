## Integrantes
[Federico Gomez](https://github.com/fededomgom)

[Damián Prámparo ](https://github.com/Dpramparo)

# Aprendizaje por Refuerzos

Repo para el trabajo practico la materia de la materia optativa Aprendizaje por Refuerzos de la Diplomatura en Ciencias de Datos, Aprendizaje
Automático y sus Aplicaciones.

## Instalación y ejecución

### Instalación desde local

#### Con pip

Pasos para instalar los paquetes específicos de RL con pip (asumiendo entorno virtual de conda con instalaciones existentes de librerías comunes como numpy, matplotlib, etc):

        pip install gymnasium
        pip install stable-baselines3[extra]  # instala las dependencias necesarias para correr el lab 2
        pip install rl_zoo3  # (Opcional) instala las dependencias para usar rl-baselines-zoo

#### Con poetry

Pasos para instalar todos los paquetes requeridos con poetry:

Instalar [poetry](https://python-poetry.org/docs/#installation):

* Desde Linux/Mac/WSL:

        pip install poetry==1.1.13

* Desde Windows:

        (Invoke-WebRequest -Uri https://install.python-poetry.org/ -UseBasicParsing).Content | python - --version 1.1.13

Comprobar que se instaló correctamente:

        poetry --version

Instalamos las dependencias (parados desde la carpeta raíz del repo):

        poetry install  # instala las dependencias necesarias
        poetry install -E zoo  # (Opcional) instala las dependencias para usar rl-baselines-zoo
        poetry install -E dev_tools  # (Opcional) instala las dependencias para usar jupyter notebooks y otras las herramientas de desarrollo

Activamos el entorno virtual:

        poetry shell

Listo! Ya podemos ejecutar los notebooks.

### Ejecución

Los notebooks están preparados para ejecutarse tanto desde localhost, como desde Google Colab.
En general, las simulaciones de estos notebooks se pueden ejecutar sin problemas desde localhost, ya que no demandan demasiados recursos computacionales (excepto si se ejecutan entrenamientos completos en entornos muy complejos, como en los de Atari).
Algunas características sólo están disponibles en localhost, como las animaciones de los agentes en los entornos.
