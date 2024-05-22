# Reproduce Results in the Paper

This repo is based on [DomainBed](https://github.com/facebookresearch/DomainBed) and [Invariance unit tests](https://github.com/facebookresearch/InvarianceUnitTests). 

## Installing requirements
```bash
conda create -n FMI python=3.9
conda activate FMI
pip install -r requirements.txt
```

## Running experiments on Colored MNIST (requires cluster)
Please follow instructions given by [DomainBed](https://github.com/facebookresearch/DomainBed) to download datasets.
```bash
cd path/to/this/repo/
export PYTHONPATH=path/to/this/repo
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher\
       --algorithms FMI --datasets ColoredMNIST\ 
       --n_hparams 1 --n_trial 10
```

After finishing the sweep, collect the result by running 
```bash
python -m domainbed.scripts.collect_results\
       --input_dir=/my/sweep/output/path
```
## Running unit tests
If one wants to run the sweep locally, use
```bash
cd path/to/this/repo
sh scripts_FMI/unit_tests/sweep.sh
```

If one is using a cluster, run 
```bash
cd path/to/this/repo/
python scripts_unit_tests/sweep.py --num_iterations 10000 --num_data_seeds 50\ 
    --num_model_seed 20 --output_dir results/ --num_samples 1000
```

After finishing the sweep, collect the result by running 
```bash
python scripts_unit_tests/collect_results.py results/COMMIT/
```

## Plot attention maps
```bash
cd path/to/this/repo
python -m domainbed.attention.plot_attention --dataset ColoredMNIST\ 
    --data_dir domainbed/data/MNIST --model FMI --test_envs 2\ 
    --model_path scripts_FMI/model/model_ColoredMNIST_FMI.pkl --row 2 --col 2
```