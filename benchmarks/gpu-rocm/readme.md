# ROCM benchmarking
Scripts under this folder are used to benchmark rocm docker for maxtext-jax with different models.
All the scripts without the _multinode suffix can be launched on single node like:
```
IMAGE="rocm/jax-maxtext-training:xxx" bash ./deepseek_v2_16b.sh
```
For the multinode scripts, they were written and tested on AMD internal cluster, and may need to be rewritten for other cluster setting. They can be launched like:
```
sbatch -N <num_nodes> llama3_70b_multinode.sh
```