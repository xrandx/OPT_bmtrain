from tqdm import auto as tqdm_lib

def greedy_generate(model, input_ids, max_seq_len, verbose=True):
    """Generate greedily from OPT.

    :param model: OPTModel
    :param input_ids: token IDs [batch_size, seq_len]
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    
    initial_input_length = input_ids.shape[1]
    current_input_ids = input_ids
    layer_past = None
    layer_past_length = 0
    all_token_ids = input_ids.tolist()
    batch_size = len(all_token_ids)

    if verbose:
        trange = tqdm_lib.trange(initial_input_length, max_seq_len)
    else:
        trange = range(initial_input_length, max_seq_len)

    for i in trange:
        input_length = current_input_ids.shape[1]
        if layer_past is not None:
            print(f"length = {i, len(layer_past)}")
        model_out, layer_past = model(
            current_input_ids,
            layer_past=layer_past,
        )
        greedy_predicted_token_ids = model_out[:, -1].argmax(-1)
        current_input_ids = greedy_predicted_token_ids[:, None]
        for i in range(batch_size):
            all_token_ids[i].append(greedy_predicted_token_ids[i])
        layer_past_length += input_length
    return all_token_ids
