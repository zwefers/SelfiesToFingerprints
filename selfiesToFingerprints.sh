#!/bin/sh
#SBATCH --account=def-jeromew
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=sefliesToFingerprints
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/logs/selfiesToFingerprints_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/logs/selfiesToFingerprints_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=10000M
python selfiesToFingerprint.py --learning_rate $1 --num_epochs $2