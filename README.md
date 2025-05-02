# rl-gen-of-kinetic-models
Generation of kinetic models using RL.


## Setting up docker container

You should be able to run the code using the provided Docker image. The image is based on the `ludekcizinsky/renaissance_with_ml` image, which is a modified version of the original Renaissance image with ML libraries added.

### Izar

First, pull the image from the Docker registry using apptainer (takes a couple of minutes):

```bash
mkdir -p /scratch/$USER/cizinsky/images
apptainer pull /scratch/izar/$USER/images/renaissance_with_ml.sif docker://ludekcizinsky/renaissance_with_ml:latest
```

Then, you can run the image with the following command:

```bash
apptainer shell --bind "$(pwd)":/home/renaissance/work /scratch/izar/$USER/images/renaissance_with_ml.sif
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy;print('Success')"
```

### Local

Assuming you have Docker desktop installed, you can run the following commands to pull the image locally (takes a couple of minutes):

```bash
docker pull ludekcizinsky/renaissance_with_ml:latest
```

Then, you can run the image with the following command:

```bash
docker run --rm -it -v "$(pwd)":/home/renaissance/work renaissance_with_ml:latest
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy;print('Success')"
```

## Running the code

Once you have the Docker container running, you can run the orirignal baseline code as follows:

```bash
cd renaissance
python 1-renaissance.py
```

However, since loading the data takes around one minute, you can execute the code in a Jupyter notebook instead or using ipython:

```bash
cd renaissance
ipython
run 1-renaissance.py
```

todo: once we have the renaissance code running, we can add info on how to run the RL code.