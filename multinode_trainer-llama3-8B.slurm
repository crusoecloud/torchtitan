#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- This script is optimized for AWS with EFA
# --- adjust NCCL_BUFFSIZE if you encounter memory
# --- constraint issues or to tune for improved performance.
# ---

#SBATCH --job-name=torchtitan_multi_node
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
# Edit topo file to suit your compute node type
export NCCL_TOPO_FILE=/etc/crusoe/nccl_topo/h100-80gb-sxm-ib-cloud-hypervisor.xml
export NCCL_SOCKET_IFNAME=eth0

# Edit lists of interfaces below to suit your compute node type - check ibstat -v on node
export NCCL_IB_HCA="mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1"
export UCX_NET_DEVICES="mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1"

export NCCL_IB_MERGE_VFS=0
export NCCL_DEBUG=INFO

export OMPI_MCA_coll_hcoll_enable=0
export PMIX_MCA_gds='^ds12'

export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_cache_${SLURM_PROCID}

export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
#export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
source .venv/bin/activate
CONFIG_FILE=${CONFIG_FILE:-"./llama3_8B.toml"}
rm -rf ./outputs
dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun --propagate=NONE torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"
dcgmi profile --resume
