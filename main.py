import torch
import os
from opt_model import OPTModel
import bmtrain as bmt
import torch
import bmtrain as bmt
from torch import nn
from transformers import AutoTokenizer, GPT2Tokenizer
import time
from tqdm import tqdm

from decode_utils import greedy_generate

# def greedy_generate_mine(model, input_ids, max_seq_len, verbose):
#     model.eval()
#     input_length = input_ids.shape[1]
#     if verbose:
#         trange = tqdm_lib.trange(input_length, max_seq_len)
#     else:
#         trange = range(input_length, max_seq_len)

#     current_input_ids = input_ids
#     for _ in trange:
#         input_length = current_input_ids.shape[1]
#         model_out, _ = model(
#             current_input_ids,
#             layer_past=None,
#         )
#         greedy_predicted_token_ids = model_out[:, -1, :].argmax(-1)
#         current_input_ids = torch.cat([current_input_ids, greedy_predicted_token_ids[:, None]], dim=-1)
#     return current_input_ids.cpu()

def infer(input_ids, opt_model, tokenizer):
    input_ids = input_ids.cuda()

    all_token_ids = greedy_generate(model=opt_model, input_ids=input_ids, max_seq_len=100, verbose=True)

    if bmt.rank() == 0:
        # output = tokenizer.batch_decode(
        #         all_token_ids, 
        #         skip_special_tokens=True, 
        #         clean_up_tokenization_spaces=True
        # )[0]
        output = tokenizer.decode(all_token_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        print(output)


def train(model, train_set):
    from static_param import save_point_path

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2, scale=128)
    dataset_len = len(train_set)

    from static_param import epoch_nums

    iter_nums = dataset_len * epoch_nums
    # lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=5e-5, warmup_iter=100, end_iter=iter_nums, num_iter=0)

    lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = 5e-5,
                                            warmup_iter = 100, 
                                            end_iter = -1,
                                            num_iter = 0)
    bmt.synchronize()

    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()

    model.train()
    for epoch in range(epoch_nums):
        print('*' * 10 + str(epoch) + '*' * 10)
        for iter, batch in enumerate(train_set):
            # load data
            torch.cuda.synchronize()

            st = time.time()
            with bmt.inspect.inspect_tensor() as inspector:
                input_ids, targets = [b.cuda() for b in batch]
                #   bsize*vocab_size*logit
                logits = model(
                    input_ids,
                    layer_past=None,
                )
                optimizer.zero_grad()
                batch, seq_len, vocab_out_size = logits.size()

                loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
            
                global_loss = bmt.sum_loss(loss).item()

                loss = optimizer.loss_scale(loss)
                loss.backward()
            
            # print inspected tensors in the forward & backward pass
            # print parameters of the model
            if iter % 100 == 0:
                bmt.print_rank(
                    bmt.inspect.format_summary(
                        inspector.get_summary()
                    )
                )
                bmt.print_rank(
                    bmt.inspect.format_summary(
                        bmt.inspect.inspect_model(model, "*")
                    )
                )
            
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, 1.0, scale = optimizer.scale, norm_type = 2)

            bmt.optim_step(optimizer, lr_scheduler)
            torch.cuda.synchronize()
            # record time and loss
            iteration_time = time.time() - st

            avg_time_recorder.record(iteration_time)
            avg_loss_recorder.record(global_loss)

            # print time and loss
            bmt.print_rank(
                "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
                    iter,
                    global_loss,
                    avg_loss_recorder.value,
                    lr_scheduler.current_lr,
                    optimizer.scale,
                    avg_time_recorder.value
                )
            )

        bmt.save(model, f"{save_point_path}/checkpoint_epoch_{epoch}.pt")


def main():

    from static_param import opt_version, opt_config, epoch_nums

    opt_model = OPTModel(opt_config, use_cache=False, dtype=torch.half)
    bmt.load(opt_model, f"/liuzyai04/tanghongjian/bmtOPT/bmtopt_weights/{opt_version}.pt")

    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(f"/liuzyai04/tanghongjian/opt/{opt_version}", use_fast=False)

    from dataprocess import get_dataloader
    train_dataset, train_wo_rat_dataset, test_dataset = get_dataloader()

    train(opt_model, train_dataset)




def test():
    from static_param import opt_version, opt_config, epoch_nums, load_point_path

    opt_model = OPTModel(opt_config, use_cache=True, dtype=torch.half)
    bmt.load(opt_model, load_point_path)

    
    bmt.load(opt_model, f"/liuzyai04/tanghongjian/bmtOPT/bmtopt_weights/{opt_version}.pt")

    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(f"/liuzyai04/tanghongjian/opt/{opt_version}", use_fast=False)
    
    token = 'Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer'
    input_ids = tokenizer.encode(token, return_tensors="pt").cuda()
    generate_ids = greedy_generate(opt_model, input_ids, 200, True)
    output = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output)


if __name__ == '__main__':
    bmt.init_distributed(
        seed=42,
        zero_level=2,
        loss_scale_factor=2,
        loss_scale_steps=100
    )
    main()
    # test()