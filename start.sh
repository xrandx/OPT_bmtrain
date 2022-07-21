
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 51002  --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost main.py



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 51002  --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost main.py



torchrun --master_port 51002  --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost main.py