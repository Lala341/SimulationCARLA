
# Simulations for Autonomous Vehicles in the CARLA environment from the extraction of patterns in video analysis of Bogotá


## Módulo de extracción de requerimientos

Para ejecutar el módulo se deben instalar las dependencias en requirements.txt y correr los siguientes comandos:

python3 vehicle_detection_main.py imshow
python3 vehicle_detection_main.py imwrite

(imshow, permite la visualización del video, imwrite devuelve el video con el análisis)

## Módulo simulación

Estos scripts deben ejecutarse dentro de la instalación de CARLA, se recomienda copiar estos scripts, junto a los resultados del módulo de extracción. Finalmente, se ejuta el simulador de CARLA, con el mapa definido y se ejecuta el servidor con:

python3 simulation.py