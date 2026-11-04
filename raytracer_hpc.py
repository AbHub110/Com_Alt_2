import numpy as np
import multiprocessing as mp
import time

# LOCALIDAD DE MEMORIA: Arreglo contiguo en C (vía NumPy) en lugar de objetos dispersos.
# Columnas: Centro X, Centro Y, Centro Z, Radio, Color R, Color G, Color B
scene = np.array([
    [0.0, 0.0, -5.0, 1.0, 255, 0, 0],
    [2.0, 0.0, -6.0, 1.0, 0, 255, 0],
    [-2.0, 0.0, -6.0, 1.0, 0, 0, 255]
], dtype=np.float32)

WIDTH = 1024
HEIGHT = 1024
TILE_SIZE = 64 # PCAM: Aglomeración (Mosaicos de 64x64 píxeles)

def procesar_mosaico(coordenadas):
    """
    Función pura que procesa un bloque independiente de la pantalla.
    Evita bloqueos (Locks/Mutex) y condiciones de carrera al no escribir en variables compartidas.
    """
    start_y, end_y, start_x, end_x = coordenadas
    
    # Buffer local temporal para el mosaico
    tile_buffer = np.zeros((end_y - start_y, end_x - start_x, 3), dtype=np.uint8)
    
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            # Matemáticas básicas de Raytracing (Normalizado a -1, 1)
            vx = (x - WIDTH/2) / (WIDTH/2)
            vy = -(y - HEIGHT/2) / (HEIGHT/2)
            vz = -1.0 
            
            mag = np.sqrt(vx*vx + vy*vy + vz*vz)
            dx, dy, dz = vx/mag, vy/mag, vz/mag
            
            # Cálculo vectorizado aprovechando la localidad de memoria de NumPy
            oc_x, oc_y, oc_z = -scene[:, 0], -scene[:, 1], -scene[:, 2]
            
            b = 2.0 * (oc_x*dx + oc_y*dy + oc_z*dz)
            c = oc_x*oc_x + oc_y*oc_y + oc_z*oc_z - scene[:, 3]*scene[:, 3]
            discriminante = b*b - 4*c
            
            hits = discriminante > 0
            if np.any(hits):
                idx = np.argmax(hits) # Tomamos la primera esfera intersectada
                tile_buffer[y - start_y, x - start_x] = scene[idx, 4:7]
                
    return coordenadas, tile_buffer

def main():
    print(f"--- Benchmarking HPC Raytracer en Python ---")
    print(f"Resolución: {WIDTH}x{HEIGHT} | Tamaño Mosaico: {TILE_SIZE}x{TILE_SIZE}")
    
    # 1. PCAM - Particionamiento y Aglomeración
    mosaicos = []
    for y in range(0, HEIGHT, TILE_SIZE):
        for x in range(0, WIDTH, TILE_SIZE):
            mosaicos.append((y, min(y+TILE_SIZE, HEIGHT), x, min(x+TILE_SIZE, WIDTH)))
    
    framebuffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    num_cores = mp.cpu_count()
    
    print(f"PCAM - Mapeo: Asignando {len(mosaicos)} mosaicos a {num_cores} núcleos lógicos...")
    
    inicio = time.time()
    
    # 2. PCAM - Concurrencia y Mapeo (Pool de Procesos esquiva el GIL de Python)
    with mp.Pool(num_cores) as pool:
        resultados = pool.map(procesar_mosaico, mosaicos)
        
    # 3. PCAM - Comunicación (Consolidación Secuencial Final)
    for coord, tile_data in resultados:
        sy, ey, sx, ex = coord
        framebuffer[sy:ey, sx:ex] = tile_data
        
    fin = time.time()
    print(f"Renderizado completado en: {fin - inicio:.4f} segundos")

if __name__ == '__main__':
    main()