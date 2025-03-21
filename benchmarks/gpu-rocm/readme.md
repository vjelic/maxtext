# ROCM benchmarking
Scripts under this folder are used to benchmark rocm docker for maxtext-jax with different models. They will launch docker and run the benchmark. **Please run them on host instead of inside any docker**

All the scripts without the _multinode suffix can be launched on single node like:
```
IMAGE="rocm/jax-maxtext-training:xxx" bash ./deepseek_v2_16b.sh
```
For the multinode one, they were written for AMD internal cluster, and will need to be adjusted for other cluster setting. They can be launched via slurm like:
```
sbatch -N <num_nodes> llama3_70b_multinode.sh
```
## Dataset
For single node scripts, they will use the $HF_HOME folder on the host. The script will mount the host HF folder to the docker. Please make sure that the data already got downloaded to $HF_HOME folder before running the script.