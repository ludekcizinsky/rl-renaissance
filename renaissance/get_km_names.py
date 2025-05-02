import pickle as pkl

path_to_kinetic_model = "models/varma_ecoli_shikki/kinetic/kin_varma_curated.yaml"
from skimpy.io.yaml import load_yaml_model
kmodel = load_yaml_model(path_to_kinetic_model)

km_list = []
for par in kmodel.parameters:
    if par.startswith('km_'):
        km_list.append(par)
n = len(km_list)
print(f'{n} km parameters found')
for i in range(10):
    print(km_list[i])

save_path = 'models/varma_ecoli_shikki/parameter_names_km_fdp1.pkl'
pkl.dump(km_list, open(save_path, 'wb'))