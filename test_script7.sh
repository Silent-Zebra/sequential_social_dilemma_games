#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=1-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
source ~/envs/causalcustom/bin/activate
python train_agents.py --num_agents 5 --num_gpus 1 --num_cpus 6 --num_workers_per_device 1 --training_iterations 50000 --env cleanup --algorithm IMPALA --intrinsic_rew_params "('ineq',0.0,0.05,1.0);('ineq',0.0,0.05,1.0);('ineq',0.0,0.05,1.0);('ineq',0.0,0.05,1.0);('ineq',0.0,0.05,1.0)"
