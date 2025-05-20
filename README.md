# rl-gen-of-kinetic-models
Generation of kinetic models using RL.


## Setting up docker container

### On node provided by Ilias 

First, pull the image from the Docker registry (this should be already done, but just in case):

```bash
sudo docker pull ludekcizinsky/renaissance_with_ml:latest
```

Next, in your home directory, create a directory for the output as well as installing the dependencies. For instance:

```bash
mkdir -p /home/rl_team/ludek/output
mkdir -p /home/rl_team/python_packages
```

Make sure that the folders are writable by the user:

```bash
chmod -R 777 /home/rl_team/ludek/output
chmod -R 777 /home/rl_team/python_packages
```

Next, visit your w&b user setting, and copy your API key. Then add it to your `.netrc` as follows:

```bash
machine api.wandb.ai
  login user
  password <your_api_key>
```

For instance, I save mine at `/home/rl_team/ludek/.netrc`.

Assuming you have cloned our rl repo and you are currently in it, you can start the docker container using the below command. Make sure to change the output directory to the one you created earlier, 
and the python packages directory to the one you created earlier (do not change the paths within the container). Finally, double check that you also mount the `.netrc` file (I include mine in the command below):

```bash
sudo docker run --rm -it -v "$(pwd)":/home/renaissance/work -v "/home/rl_team/ludek/output:/home/renaissance/output" -v "/home/rl_team/python_packages:/home/renaissance/.local" -v "/home/rl_team/ludek/.netrc:/home/renaissance/.netrc" ludekcizinsky/renaissance_with_ml
```

Finally, since we have to work with python 3.6, we have to downgrade weights and biases:

```bash
pip install wandb==0.15.11
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy; import torch;print('Success')"
```

### Izar (if you have access, then recommended)

First, pull the image from the Docker registry using apptainer (takes a couple of minutes):

```bash
mkdir -p /scratch/izar/$USER/images
apptainer pull /scratch/izar/$USER/images/renaissance_with_ml.sif docker://ludekcizinsky/renaissance_with_ml:latest
```

Then, you can run the image with the following command:

```bash
mkdir -p /scratch/izar/$USER/rl-for-kinetics/output
apptainer shell --nv --bind "$(pwd)":/home/renaissance/work --bind "/scratch/izar/$USER/rl-for-kinetics/output:/home/renaissance/output" /scratch/izar/$USER/images/renaissance_with_ml.sif
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy; import torch;print('Success')"
```

Finally, note that you can use GPU to speed up the training. In the config, set `device: cuda`.

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

### Running with SLURM

For those who have access to Izar, you can run the training with SLURM as follows. First, checkout the `train.slurm` file and adjust it as needed, usually you should change the name of the job, time requested and possibly also the account. Then, specify which configuration you want to use by changing the script at the bottom of the file (see the example usage in the file). Finally, you can submit the job with the following command (change to your izar usernmae):

```bash
sbatch train.slurm cizinsky
```

If things go wrong, you can check the output in the `outputs/slurm` directory.
