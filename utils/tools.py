import torch.distributed as dist

#world_size卡数
def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()#返回tensor的拷贝
    dist.all_reduce(tensor, op)#分布式求和
    tensor.div_(world_size)
    return tensor
