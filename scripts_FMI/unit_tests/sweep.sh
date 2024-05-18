#!/bin/bash
cd ../../
python scripts_unit_tests/sweep.py --num_iterations 10000 --num_data_seeds 50 --num_model_seed 20 --output_dir results/ --num_samples 1000
