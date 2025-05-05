import hydra
from omegaconf import DictConfig, OmegaConf

from renaissance.kinetics.jacobian_solver import check_jacobian


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    print("-" * 50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-" * 50)


    # Call solvers from SKimPy
    chk_jcbn = check_jacobian()

    # Integrate data
    met_model_path = cfg.paths.met_model_path
    chk_jcbn._load_ktmodels(met_model_path, 'fdp1')           # Load kinetic and thermodynamic data
    chk_jcbn._load_ssprofile(met_model_path, 'fdp1', cfg.constraints.ss_idx)  # Integrate steady state information

    # TODO: here goes our training loop using whatever model

if __name__ == "__main__":
    train()
