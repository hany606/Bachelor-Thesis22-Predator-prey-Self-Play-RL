import torch

policy_arch_dict = {
    "None":None,
    "drone_ppo": dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]),
    "drone_sac": dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512, 256, 128]) # or None # or dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))
}

def get_policy_arch(arch_str):
    return policy_arch_dict.get(arch_str, None)