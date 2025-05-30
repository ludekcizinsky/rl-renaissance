# RL-Renaissance

TODO: please add method overview figure here.
TODO: please add abstract here.


## 🔮 Future work

For those who are interested in the future of this project, here are some ideas:

- Try using more predictions per step to obtain better estimate of the distribution of the next state, and stabilise the training
- Run for more than 100 episodes, e.g. 150, 200
- Try integrating timestep information into the model
- Try using more complex model such as LSTM that can take into account temporal information
- Try using non-stochastic policy, however stochastic input

## 🛠️ Setting up the environment

In order to run the code, you need to set up the **docker container** which is due to the dependency on [skimpy](https://github.com/EPFL-LCSB/skimpy) package. To make your life easier, and not having to build the container yourself, it is publicly [available](https://hub.docker.com/repository/docker/ludekcizinsky/renaissance_with_ml/general) on Docker Hub.

### 👌 Izar or any other SLURM cluster that supports apptainer (if you have access, then recommended)

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

For some random reason, when inside the container, you also need to run:

```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

If things go well, you should be able to execute the following command to check if the image is working:

```bash
python -c "import skimpy; import torch;print('Success')"
```

### 🌿 Laboratory of Computational Systems Biotechnology (LCSB)

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

## 📊 Inference

We save all our final results in the `wandb` platform which is publicly [accessible](https://wandb.ai/ludekcizinsky/rl-renaissance/table?nw=4uf48ytfjog). You can take a look there to download any trained model using the available tags, please see how to do so in [this notebook](notebooks/download_trained_models.ipynb). However, if you just want to use the best model, we have saved all neccesary files in the `inference` directory under [best model folder](inference/best_model).

Finally, to run it and play around with the generated kinetic models, and inspect the rewards, you can use the [inference notebook](notebooks/inference.ipynb).

## 🚀 Reproducing the results

Assuming you are now inside the docker container and followed the instructions above, you can now run the code in the following way.

In order to reproduce the results of the **main method** reported in the report, you can run the following command:

```bash
python train.py
```

In order to reproduce the results of the **ablation studies** reported in the report, you can schedule a SLURM job as follows:

```bash
sbatch train1.slurm <your_username>
sbatch train2.slurm <your_username>
```

Or just look at the `train1.slurm` and `train2.slurm` files for which configurations you want to run on whatever system you are on.