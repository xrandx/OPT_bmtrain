import os
import torch
map_to_bmt = {
    "decoder.embed_tokens.weight": "embed_tokens.weight",
    "decoder.embed_positions.weight": "embed_positions.weight",
    "decoder.final_layer_norm.weight": "final_layernorm.weight",
    "decoder.final_layer_norm.bias": "final_layernorm.bias"
}

part2bmt = {
    "self_attn.k_proj.weight": "attention.k_proj.weight", 
    "self_attn.k_proj.bias": "attention.k_proj.bias", 
    "self_attn.v_proj.weight": "attention.v_proj.weight", 
    "self_attn.v_proj.bias": "attention.v_proj.bias", 
    "self_attn.q_proj.weight": "attention.q_proj.weight", 
    "self_attn.q_proj.bias": "attention.q_proj.bias", 
    "self_attn.out_proj.weight": "attention.out_proj.weight", 
    "self_attn.out_proj.bias": "attention.out_proj.bias", 
    "self_attn_layer_norm.weight": "input_layernorm.weight", 
    "self_attn_layer_norm.bias": "input_layernorm.bias", 
    "fc1.weight": "mlp.dense_h_to_4h.weight", 
    "fc1.bias": "mlp.dense_h_to_4h.bias", 
    "fc2.weight": "mlp.dense_4h_to_h.weight", 
    "fc2.bias": "mlp.dense_4h_to_h.bias", 
    "final_layer_norm.weight": "post_attention_layernorm.weight", 
    "final_layer_norm.bias": "post_attention_layernorm.bias", 
}

def convert(src, dst):
    state = dict()
    for fname in os.listdir(src):
        if fname[-4:] == ".bin":
            st = torch.load(os.path.join(src, fname), map_location=torch.device("cpu"))
            state = {**st, **state}

    new_state = dict()

    for k, v in state.items():
        new_key = ""
        if k in map_to_bmt:
            new_key = map_to_bmt[k]
        else:
            k = k.replace("decoder.layers", "layer_list")
            for name in part2bmt:
                if name in k:
                    new_key = k.replace(name, part2bmt[name])
                    break
        new_state[new_key] = v #   test

    print(dst)
    torch.save(new_state, dst)

if __name__ == "__main__":
    convert(
        src="/liuzyai04/tanghongjian/opt/13b",
        dst="/liuzyai04/tanghongjian/bmtOPT/bmtopt_weights/13b.pt"
    )

