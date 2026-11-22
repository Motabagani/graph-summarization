# Copy to config.sh (git-ignored), edit, then `source config.sh` before submitting any job.
# sbatch passes these environment variables on to the jobs.

export DATA_ROOT=/scratch/$USER/NLP/datasets      # where the six datasets live (see slurm/download_datasets.sh)
export RESULTS_DIR=$PWD/results                    # where runs write predictions/metrics (default: ./results)
export VENV=/scratch/$USER/envs/dl_env             # python virtualenv to activate in jobs (leave empty to skip)
export NLTK_DATA=/scratch/$USER/nltk_data          # NLTK tokenizer data (downloaded automatically)
export SBATCH_ACCOUNT=your_slurm_account           # Slurm reads this for every sbatch (e.g. torch_pr_xxx_general)
# export SBATCH_PARTITION=...                      # optional
