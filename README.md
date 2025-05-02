# rl-gen-of-kinetic-models
Generation of kinetic models using RL.


## Installation

### Izar

This guide assumes you have access to the Izar compute cluster. First, clone this repository in your home directory. Then, pull the skimpy docker image:

```bash
mkdir -p /scratch/izar/$USER/images
apptainer pull /scratch/izar/$USER/images/skimpy.sif docker://danielweilandt/skimpy
```

Next, install skimpy:

```bash
module load git-lfs
git clone git@github.com:EPFL-LCSB/skimpy.git /home/$USER/skimpy
git -C /home/$USER//skimpy lfs pull
```

Next, get inside the container, mount the code and data directory, and install the requirements:

```bash
mkdir -p /scratch/izar/$USER/rl-for-kinetics/data
apptainer shell \
  --bind /scratch/izar/$USER/rl-for-kinetics/data:/mnt/data \
  --bind /home/$USER/skimpy:/mnt/skimpy \
  --bind /home/$USER/rl-gen-of-kinetic-models:/mnt/code \
  /scratch/izar/$USER/images/skimpy.sif
```




Then, inside the Apptainer container, run the following commands to install the requirements:

```bash
pip install -r /mnt/code/requirements.txt
```
