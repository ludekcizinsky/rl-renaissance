# rl-gen-of-kinetic-models
Generation of kinetic models using RL.


## Setting up docker container

### On node provided by Ilias (recommended)

First, pull the image from the Docker registry (this should be already done, but just in case):

```bash
sudo docker pull ludekcizinsky/renaissance_with_ml:latest
```

Next, in your home directory, create a directory for the output. For instance:

```bash
mkdir -p /home/rl_team/ludek/output
```

Make sure that the folder is writable by the user:

```bash
chmod -R 777 /home/rl_team/ludek/output
```

Assuming you have cloned our rl repo and you are currently in it, you can start the docker container using the below command (make sure to change the output directory to the one you created earlier):

```bash
sudo docker run --rm -it -v "$(pwd)":/home/renaissance/work -v "/home/rl_team/ludek/output:/home/renaissance/output" ludekcizinsky/renaissance_with_ml
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy; import torch;print('Success')"
```

### Izar 

First, pull the image from the Docker registry using apptainer (takes a couple of minutes):

```bash
mkdir -p /scratch/izar/$USER/images
apptainer pull /scratch/izar/$USER/images/renaissance_with_ml.sif docker://ludekcizinsky/renaissance_with_ml:latest
```

Then, you can run the image with the following command:

```bash
mkdir -p /scratch/izar/$USER/rl-for-kinetics/output
apptainer shell --nv --bind "$(pwd)":/home/renaissance/work --bind "/scratch/izar/$USER/rl-for-kinetics/output:/home/renaissance/output" /scratch/izar/$USER/images/renaissance_with_ml.sif
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy; import torch;print('Success')"
```

Finally, note that you can use GPU to speed up the training. In the config, set `device: cuda`.

### On node provided by Ilias

Assuming you have Docker desktop installed, you can run the following commands to pull the image locally (takes a couple of minutes):

```bash
docker pull ludekcizinsky/renaissance_with_ml:latest
```

Then, start the docker container using the below command (make sure to change the path `path_to_local_output` accordingly):

```bash
sudo docker run --rm -it -u $(id -u):$(id -g) -v "$(pwd)":/home/renaissance/work ludekcizinsky/renaissance_with_ml
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy; import torch;print('Success')"
```

## Running the code

### Baseline code

Once you have the Docker container running, you can run the original baseline code as follows:

```bash
cd renaissance
python 1-renaissance.py
```

### RL code

To run the training with default configuration (see `configs/train.yaml`), you can use the following command:

```bash
python train.py
```

In practice, however, you want to experiment with different configurations. You can override the default configuration by using the following command:

```bash
python train.py method.actor_lr=1e-4 method.latent_dim=256
```


