#!/bin/sh
#SBATCH --account=def-jeromew
#SBATCH --time=00:03:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sefliesToFingerprints
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/logs/selfiesToFingerprints_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/logs/selfiesToFingerprints_%A.err
#SBATCH --mem=1M
python out_test.py