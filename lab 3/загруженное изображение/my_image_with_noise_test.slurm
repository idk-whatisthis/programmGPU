#!/bin/bash
#SBATCH --job-name=my_image_with_noise_test
#SBATCH --partition=gpuserv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=output_my_image_with_noise_%j.log

echo "=== My Image Test with Noise Program ==="
module load nvidia/cuda-11.7

echo "=== Running the noise version with my image ==="
./final_technical_with_noise my_image.png

echo "=== Checking results ==="
ls -la technical_with_noise_*.png
file technical_with_noise_*.png
ls -lh technical_with_noise_*.png

echo "=== Job completed ==="
