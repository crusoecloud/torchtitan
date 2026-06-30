# Running TorchTitan on Crusoe Cloud's Terraform SLURM Solution or Crusoe Managed Slurm #

Forked from the TorchTitan repo (https://github.com/pytorch/torchtitan).  
Default branch is release-v0.2.2 which is the latest release that still supports the .toml config file method (easiest for our use case).

*Torchtitan is a PyTorch native platform designed for rapid experimentation and large-scale training of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, torchtitan provides a flexible foundation for developers to build upon. With torchtitan extension points, one can easily create custom extensions tailored to specific needs.*

We will do multinode fine tuning of Llama3_8b and Llama3_70b using the C4 dataset. The results section at the bottom of the page will be extended to cover multiple GPUs and cluster and batch sizes. The dataset can be downloaded during training iteself, or pre-downloaded to local filesystem/NFS, or saved in Crusoe Object St

## Prerequites

We expect the following to be set up as part of your environment:
- Terraform Slurm solution on Crusoe: https://github.com/crusoecloud/slurm, or:
- Crusoe Managed Slurm (on Kubernetes): https://docs.crusoecloud.com/orchestration/slurm/overview 

Other pre-requisites include:
- We expect to have a Slurm cluster set up with compute nodes using any of the GPU VMs provided by Crusoe (but as shown in the results section, most of our benchmarking is done on the following: H100, H200, B200, and GB200)
- Shared home and/or data directories mounted to all the nodes, where you can clone the repo and install dependencies
- If desired, Crusoe object storage
- HuggingFace account with access to Llama 3.1 8B model

## Setup 

Please follow the setup for the specific GPU compute node types as shown below:

1. Clone the torchtitan repository and cd into the cloned directory

```
git clone --branch release-v0.2.2 --depth 1 https://github.com/crusoecloud/torchtitan.git
cd torchtitan
```

2. Create and activate a Python virtual environment, then install dependencies. **For GB200, create a Python virtual environment on one of the compute nodes (and not the login or head node)**

```
# install uv (uv is pre-installed on Crusoe Managed Slurm)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

4. Download Llama 3.1 8B and 70B tokenizers from Huggingface

```
#Download tokenizer from HuggingFace
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=<your huggingface token>
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-70B --assets tokenizer --hf_token=<your huggingface token>
```
                                
### Running the job ###

1. Edit your chosen multinode_trainer_llama3-*.slurm file (the 8B or 70B version) to set the correct number of nodes, topology file etc for your cluster. 

** For quicker performance on repeated training runs: download C4 data set to a shared volume on your cluster **
```
cd /data
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
#this next step takes about an hour
git lfs pull --include "en/*"
```
Update the .toml training config files with * dataset_path = "/data/c4/" * in the \[training\] section

**For testing with Crusoe Object Storage**

Follow the steps above to create a local download of the C4 dataset tar.gz files, then copy those files into your Crusoe object storage bucket. Update dataset_path in the .toml training config files to be the bucket containing the files e.g s3://my-c4-bucket/. Provide an S3_ENDPOINT_URL environment variable set to the correct object storage URL of the Crusoe Cloud region where you are testing (set to https://object.us-east1-a.crusoecloudcompute.com by default). S3 credentials and config are read from the standard location ~/.aws/credentials and ~/.aws/config or as environment variables.

Run the job from the slurm login node: `sbatch multinode_trainer.sbatch`. Run `watch squeue` to see that the job is running - if it doesn’t go to `R` status it could be that you didn’t have sufficient nodes in `idle` state, or that you request resources that no node has (e.g too many GPU or CPU per node)

When the job is running, its logs will be written to `slurm-x.out` where `x` is the ID of the slurm job. The outputs below show the performance to be expected on a cluster where all the nodes are in the same InfiniBand Network or IMEX partition (for GB200).

# Torchtitan performance results for various cluster configs #

|GPU Type|NVIDIA Driver/CUDA|Compute nodes in training job|Batch size|Performance indicators at step 100 of Torchtitan test described on this page|
|--------|------------------|-----------|----------|-----------------------------------------------------------------------------|
|GB200   |580/CUDA 13.0|17 (68 GPU)                  |5         |step: 100  loss:  6.4277  grad_norm:  2.9062  memory: 158.61GiB(86.20%)  tps: 17,581  tflops: 1,018.20  mfu: 45.25%|
|GB200   |580/CUDA 13.0|9 (36 GPU)                   |5         |step: 100  loss:  8.2954  grad_norm: 29.3788  memory: 159.31GiB(86.58%)  tps: 17,975  tflops: 1,040.99  mfu: 46.27%|
|GB200   |580/CUDA 13.0|3 (12 GPU)                   |5         |step: 100  loss:  6.5572  grad_norm:  3.6680  memory: 161.65GiB(87.85%)  tps: 18,284  tflops: 1,058.93  mfu: 47.06%|
|B200    |580/CUDA 13.0|2 (16 GPU)                   |5         |step: 100  loss:  6.1751  grad_norm:  1.4532  memory: 1cl61.51GiB(90.56%)  tps: 17,500  tflops: 1,013.53  mfu: 45.05%|
|B200    |580/CUDA 13.0|1 (8 GPU)                    |5         |step: 100  loss:  6.3314  grad_norm:  4.0237  memory: 165.16GiB(92.60%)  tps: 17,469  tflops: 1,011.68  mfu: 44.96% |
|H200    |570/CUDA 12.8|16 (128 GPU)                 |4         |step: 100  loss:  7.2984  grad_norm:  5.6297  memory: 128.17GiB(91.66%)  tps: 9,044  tflops: 523.76  mfu: 52.96% |
|H200    |570/CUDA 12.8|8 (64 GPU)                   |4         |step: 100  loss:  6.14398  grad_norm:  1.8245  memory: 128.79GiB(92.11%)  tps: 8,844  tflops: 512.21  mfu: 51.79%|
|H200    |570/CUDA 12.8|4 (32 GPU)                   |4         |step: 100  loss:  6.15755  grad_norm:  1.5348  memory: 129.80GiB(92.84%)  tps: 8,895  tflops: 515.14  mfu: 52.09% |
|H100    |570/CUDA 12.8|16 (128 GPU)                 |2         |step: 100  loss:  6.3626  grad_norm:  5.2285  memory: 66.38GiB(83.83%)  tps: 7,169  tflops: 415.21  mfu: 41.98% |
|H100    |570/CUDA 12.8|4 (32 GPU)                   |2         |step: 100  loss:  6.35632  grad_norm:  4.4844  memory: 67.96GiB(85.82%)  tps: 7,582  tflops: 439.14  mfu: 44.40% |
|H100    |580/CUDA 13.0| 2 (16 GPU)                  |2         |step: 100  loss:  6.47847  grad_norm:  2.8199  memory: 69.85GiB(88.21%)  tps: 8,714  tflops: 504.69  mfu: 51.03% |

### When stress testing nodes and looking for issues, we are looking for consistent results across all GPUS plus high memory utilization and MFU ###

<img width="1164" height="179" alt="image" src="https://github.com/user-attachments/assets/73c405ca-2fb5-491b-b443-b3649182a248" />

# Small-scale GB200 vs B200 performance comparison for fine-tuning Llama3.1-70B with TorchTitan #

Cluster setups: 2 x B200.8x node vs 4 x GB200.4x nodes; C4 training dataset stored on the same shared volume.  
Wall clock time for 1000 steps:  
B200: 40 mins 58 sec  
GB200: 48 minutes 42 sec  
