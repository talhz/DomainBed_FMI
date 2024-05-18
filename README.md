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
python -m domainbed.scripts.sweep launch\
       --data_dir=domainbed/data/MNIST \
       --output_dir=sweep_out \
       --command_launcher your_launcher --skip_confirmation --algorithms FMI --datasets ColoredMNIST \ 
       --n_hparams 1 --n_trial 10
```

## Running unit tests
```bash
sh scripts_FMI/unit_tests/sweep.sh
```

## Plot attention maps
```bash
cd path/to/this/repo
python -m domainbed.attention.plot_attention --dataset ColoredMNIST \ 
    --data_dir domainbed/data/MNIST --model FMI --test_envs 2 \ 
    --model_path scripts_FMI/model/model_ColoredMNIST_FMI.pkl --row 2 --col 2
```