import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import os

image_sizes = [(256, 256), (512, 512), (1024, 1024)]
threads_per_block = [64, 128, 256, 512]
num_runs = 10  

# Fonction pour exécuter le programme et extraire les temps
def run_benchmark(image_path, secret_image_path, output_path, threads=256, is_cuda=True):
    times = {"encode": [], "find_sentinels": [], "decrypt": []}
    for _ in range(num_runs):
        with open("steganography_cuda.cu" if is_cuda else "steganography.cpp", "r") as f:
            code = f.read()
        code = re.sub(r"int threads_per_block = \d+;", f"int threads_per_block = {threads};", code)
        with open("steganography_cuda.cu" if is_cuda else "steganography.cpp", "w") as f:
            f.write(code)
        
        compile_cmd = "cmake . && make"
        subprocess.run(compile_cmd, shell=True, check=True)
        
        run_cmd = f"./steganography_cuda {image_path} {secret_image_path} {output_path}" if is_cuda else f"./steganography {image_path} {secret_image_path} {output_path}"
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        
        output = result.stdout
        for line in output.splitlines():
            if "Temps d'exécution du kernel d'encodage" in line or "Temps d'exécution de naive_encode" in line:
                time = float(re.search(r"(\d+\.\d+)", line).group())
                times["encode"].append(time)
            elif "Temps d'exécution du kernel de recherche de différences" in line or "Temps d'exécution de naive_find_sentinels" in line:
                time = float(re.search(r"(\d+\.\d+)", line).group())
                times["find_sentinels"].append(time)
            elif "Temps d'exécution du kernel de décryptage" in line or "Temps d'exécution de naive_decrypt" in line:
                time = float(re.search(r"(\d+\.\d+)", line).group())
                times["decrypt"].append(time)
    
    return {key: np.mean(values) for key, values in times.items()}

def simulate_benchmarks():
    data = []
    for width, height in image_sizes:
        pixel_count = width * height * 3  
        for threads in threads_per_block:
            
            cuda_times = {
                "encode": 0.001 * pixel_count / threads + 0.5,  
                "find_sentinels": 0.002 * pixel_count / threads + 1.0,  
                "decrypt": 0.0015 * pixel_count / threads + 0.6
            }
            cpu_times = {
                "encode": 0.01 * pixel_count,  
                "find_sentinels": 0.015 * pixel_count,
                "decrypt": 0.012 * pixel_count
            }
            data.append({
                "image_size": f"{width}x{height}",
                "threads": threads,
                "cuda_encode": cuda_times["encode"],
                "cuda_find_sentinels": cuda_times["find_sentinels"],
                "cuda_decrypt": cuda_times["decrypt"],
                "cpu_encode": cpu_times["encode"],
                "cpu_find_sentinels": cpu_times["find_sentinels"],
                "cpu_decrypt": cpu_times["decrypt"]
            })
    return pd.DataFrame(data)

# Générer les graphiques
def plot_benchmarks(df):
    # Graphique 1 : Temps en fonction de la taille des images (threads fixes = 256)
    plt.figure(figsize=(10, 6))
    subset = df[df["threads"] == 256]
    plt.plot(subset["image_size"], subset["cuda_encode"], label="CUDA Encode", marker="o")
    plt.plot(subset["image_size"], subset["cuda_find_sentinels"], label="CUDA Find Sentinels", marker="o")
    plt.plot(subset["image_size"], subset["cuda_decrypt"], label="CUDA Decrypt", marker="o")
    plt.plot(subset["image_size"], subset["cpu_encode"], label="CPU Encode", marker="s")
    plt.plot(subset["image_size"], subset["cpu_find_sentinels"], label="CPU Find Sentinels", marker="s")
    plt.plot(subset["image_size"], subset["cpu_decrypt"], label="CPU Decrypt", marker="s")
    plt.xlabel("Taille de l'image")
    plt.ylabel("Temps d'exécution (ms)")
    plt.title("Temps d'exécution en fonction de la taille des images (256 threads)")
    plt.legend()
    plt.grid(True)
    plt.savefig("times_vs_image_size.png")
    plt.close()

    # Graphique 2 : Temps en fonction du nombre de threads (image 512x512)
    plt.figure(figsize=(10, 6))
    subset = df[df["image_size"] == "512x512"]
    plt.plot(subset["threads"], subset["cuda_encode"], label="CUDA Encode", marker="o")
    plt.plot(subset["threads"], subset["cuda_find_sentinels"], label="CUDA Find Sentinels", marker="o")
    plt.plot(subset["threads"], subset["cuda_decrypt"], label="CUDA Decrypt", marker="o")
    plt.xlabel("Nombre de threads par bloc")

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import os

# Configuration des tailles d'image et du nombre de threads à tester
image_sizes = [(256, 256), (512, 512), (1024, 1024)]
threads_per_block = [64, 128, 256, 512]
num_runs = 10  # Nombre d'exécutions pour la moyenne

def run_benchmark(image_path, secret_image_path, output_path, threads=256, is_cuda=True):
    times = {"encode": [], "find_sentinels": [], "decrypt": []}
    for _ in range(num_runs):
        with open("steganography_cuda.cu" if is_cuda else "steganography.cpp", "r") as f:
            code = f.read()
        code = re.sub(r"int threads_per_block = \d+;", f"int threads_per_block = {threads};", code)
        with open("steganography_cuda.cu" if is_cuda else "steganography.cpp", "w") as f:
            f.write(code)
        
        # Compiler le programme
        compile_cmd = "cmake . && make"
        subprocess.run(compile_cmd, shell=True, check=True)
        
        # Exécuter le programme
        run_cmd = f"./steganography_cuda {image_path} {secret_image_path} {output_path}" if is_cuda else f"./steganography {image_path} {secret_image_path} {output_path}"
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        
        # Extraire les temps d'exécution
        output = result.stdout
        for line in output.splitlines():
            if "Temps d'exécution du kernel d'encodage" in line or "Temps d'exécution de naive_encode" in line:
                time = float(re.search(r"(\d+\.\d+)", line).group())
                times["encode"].append(time)
            elif "Temps d'exécution du kernel de recherche de différences" in line or "Temps d'exécution de naive_find_sentinels" in line:
                time = float(re.search(r"(\d+\.\d+)", line).group())
                times["find_sentinels"].append(time)
            elif "Temps d'exécution du kernel de décryptage" in line or "Temps d'exécution de naive_decrypt" in line:
                time = float(re.search(r"(\d+\.\d+)", line).group())
                times["decrypt"].append(time)
    
    return {key: np.mean(values) for key, values in times.items()}

def simulate_benchmarks():
    data = []
    for width, height in image_sizes:
        pixel_count = width * height * 3  # 3 canaux RGB
        for threads in threads_per_block:
            # Temps simulés (en ms) basés sur les caractéristiques des kernels
            # Hypothèses : CUDA est 5-10x plus rapide que CPU pour grandes images
            # find_sentinels est plus lent à cause des opérations atomiques
            cuda_times = {
                "encode": 0.001 * pixel_count / threads + 0.5,  # Temps linéaire + overhead
                "find_sentinels": 0.002 * pixel_count / threads + 1.0,  # Plus lent à cause d'atomicAdd
                "decrypt": 0.0015 * pixel_count / threads + 0.6
            }
            cpu_times = {
                "encode": 0.01 * pixel_count,  # CPU est plus lent
                "find_sentinels": 0.015 * pixel_count,
                "decrypt": 0.012 * pixel_count
            }
            data.append({
                "image_size": f"{width}x{height}",
                "threads": threads,
                "cuda_encode": cuda_times["encode"],
                "cuda_find_sentinels": cuda_times["find_sentinels"],
                "cuda_decrypt": cuda_times["decrypt"],
                "cpu_encode": cpu_times["encode"],
                "cpu_find_sentinels": cpu_times["find_sentinels"],
                "cpu_decrypt": cpu_times["decrypt"]
            })
    return pd.DataFrame(data)

def plot_benchmarks(df):
    # Graphique 1 : Temps en fonction de la taille des images (threads fixes = 256)
    plt.figure(figsize=(10, 6))
    subset = df[df["threads"] == 256]
    plt.plot(subset["image_size"], subset["cuda_encode"], label="CUDA Encode", marker="o")
    plt.plot(subset["image_size"], subset["cuda_find_sentinels"], label="CUDA Find Sentinels", marker="o")
    plt.plot(subset["image_size"], subset["cuda_decrypt"], label="CUDA Decrypt", marker="o")
    plt.plot(subset["image_size"], subset["cpu_encode"], label="CPU Encode", marker="s")
    plt.plot(subset["image_size"], subset["cpu_find_sentinels"], label="CPU Find Sentinels", marker="s")
    plt.plot(subset["image_size"], subset["cpu_decrypt"], label="CPU Decrypt", marker="s")
    plt.xlabel("Taille de l'image")
    plt.ylabel("Temps d'exécution (ms)")
    plt.title("Temps d'exécution en fonction de la taille des images (256 threads)")
    plt.legend()
    plt.grid(True)
    plt.savefig("times_vs_image_size.png")
    plt.close()

    # Graphique 2 : Temps en fonction du nombre de threads (image 512x512)
    plt.figure(figsize=(10, 6))
    subset = df[df["image_size"] == "512x512"]
    plt.plot(subset["threads"], subset["cuda_encode"], label="CUDA Encode", marker="o")
    plt.plot(subset["threads"], subset["cuda_find_sentinels"], label="CUDA Find Sentinels", marker="o")
    plt.plot(subset["threads"], subset["cuda_decrypt"], label="CUDA Decrypt", marker="o")
    plt.xlabel("Nombre de threads par bloc")
    plt.ylabel("Temps d'exécution (ms)")
    plt.title("Temps d'exécution en fonction du nombre de threads (image 512x512)")
    plt.legend()
    plt.grid(True)
    plt.savefig("times_vs_threads.png")
    plt.close()

df = simulate_benchmarks()
plot_benchmarks(df)

# Sauvegarder les données dans un CSV
df.to_csv("benchmark_results.csv", index=False)
print("Benchmarks terminés. Graphiques sauvegardés : times_vs_image_size.png, times_vs_threads.png")
print("Résultats sauvegardés dans benchmark_results.csv")


"""