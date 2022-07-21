import io_utils
import os
import re
import json
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import random
import bmtrain as bmt


from static_param import opt_version
tokenizer = AutoTokenizer.from_pretrained(f"/liuzyai04/tanghongjian/opt/{opt_version}", use_fast=False)



def get_gsm8k():
    save_to = "/liuzyai04/tanghongjian/finetune_opt/dataset/"

    def get_res(ds):
        res = []
        for item in ds:
            idx = item["answer"].index("####")
            temp = {
                "question": f'{item["question"]}',
                "rationale": item["answer"][:idx].strip(),
                "answer": item["answer"][idx:]
            }
            res.append(temp)
        return res

    train = io_utils.read_jsonl("/liuzyai04/tanghongjian/CoT/data/grade_school_math/data/train.jsonl")
    train = get_res(train)
    io_utils.save_jsonl(train, save_to + "grade_school_math_train.jsonl")

    test = io_utils.read_jsonl("/liuzyai04/tanghongjian/CoT/data/grade_school_math/data/test.jsonl")
    test = get_res(get_res(test))
    io_utils.save_jsonl(test, save_to + "grade_school_math_test.jsonl")


def train_collate_fn_wo_rat(batch):
    res = []
    for mini_batch in batch:
        token = "Question: " + mini_batch["question"] + "\nAnswer: \n" + mini_batch["answer"] + "</s>"
        res.append(token)

    res = tokenizer.batch_encode_plus(res, padding=True, return_tensors="pt")
    return res[:, :-1], res[:, 1:]


def test_collate_fn(batch):
    res = []
    qs = []
    ans = []
    for mini_batch in batch:
        token = "Question: " + mini_batch["question"] + "\nAnswer: "
        res.append(token)
        qs.append(mini_batch["question"])
        ans.append(mini_batch["answer"])
    res = tokenizer.batch_encode_plus(res, padding=True, return_tensors="pt")
    return res.input_ids, qs, ans


def train_collate_fn(batch):
    res = []
    for mini_batch in batch:
        token = "Question: " + mini_batch["question"] + "\nAnswer: " + mini_batch["rationale"] + "\n" + mini_batch["answer"]
        res.append(token)

    res = tokenizer.batch_encode_plus(res, padding=True, return_tensors="pt").input_ids
    return res[:, :-1], res[:, 1:]


def get_dataloader():
    from static_param import batch_size

    train_dataset = DstributedDataset(
        corpus_path="/liuzyai04/tanghongjian/bmtOPT/dataset/grade_school_math_test.jsonl", batch_size=batch_size, world_size=bmt.world_size(), rank=bmt.rank(), shuffle=True, collate_fn=train_collate_fn
    )

    train_wo_rat_dataset = DstributedDataset(
        corpus_path="/liuzyai04/tanghongjian/bmtOPT/dataset/grade_school_math_train.jsonl", batch_size=batch_size, world_size=bmt.world_size(), rank=bmt.rank(), shuffle=True, collate_fn=train_collate_fn_wo_rat
    )

    test_dataset = DstributedDataset(
        corpus_path="/liuzyai04/tanghongjian/bmtOPT/dataset/grade_school_math_test.jsonl", batch_size=1, world_size=bmt.world_size(), rank=bmt.rank(), shuffle=False, collate_fn=test_collate_fn
    )

    # test_src = '/liuzyai04/tanghongjian/finetune_opt/dataset/our_new_gsm8k0.jsonl'
    return train_dataset, train_wo_rat_dataset, test_dataset


class DstributedDataset:
    def __init__(self, corpus_path, batch_size, world_size, rank, shuffle, collate_fn):
        super(DstributedDataset, self).__init__()
        self.corpus_path = corpus_path
        self.all_data = []
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        
        self.collate_fn = collate_fn
        self.rank = rank
        self.indexes = []
        self.index_size = 0
        self._create_train_data()


    def _create_train_data(self):
        self.all_data = io_utils.read_jsonl(self.corpus_path)
        self.length = len(self.all_data)
        for idx in range(0, len(self.all_data), self.world_size):
            idx = idx + self.rank
            if idx < self.length:
                self.indexes.append(idx)
        batch_split = []
        temp = []
        for idx in self.indexes:
            temp.append(idx)
            if len(temp) == self.batch_size:
                batch_split.append(temp[:])
                temp.clear()
        self.indexes = batch_split
        self.index_size = len(self.indexes)

    def __getitem__(self, index):
        idxes = self.indexes[index]
        mini_batch = [self.all_data[i] for i in idxes]
        res = self.collate_fn(mini_batch)
        if index == self.index_size - 1 and self.shuffle:
            random.shuffle(self.indexes)
        return res

    def __len__(self):
        return self.index_size


def filter_right_ans(root = "/liuzyai04/tanghongjian/finetune_opt/output/wo_rat"):
    REG = r"(#{4}\s+\d+)"

    def get_true_ans(ans):
        right_ans = []
        for item in ans:
            temp = {
                "question": item['question'],
                "answer": item["answer"]
            }
            # "our_ans-default", "our_ans-t=0.7_k=40", 
            for param in ["our_ans-p=0.9"]:
                res = re.findall(REG, item[param])
                if item["answer"] in res:
                    temp[param] = item[param]
            if len(temp) > 2:
                right_ans.append(temp)
        print(len(right_ans))
        return right_ans


    for fname in os.listdir(root):
        file_path = os.path.join(root, fname)
        file = io_utils.read_json(file_path)
        true_ans = get_true_ans(file)
        io_utils.save_json(true_ans, os.path.join(root, f"true_{fname}"))


def convert2xlsx(df, dst):
    writer = pd.ExcelWriter(dst, engine='xlsxwriter')
    df.to_excel(writer, sheet_name="sheet1", index=False)
    workbook  = writer.book
    worksheet = writer.sheets["sheet1"]
    wrap_format = workbook.add_format({
        'text_wrap': True,
        'align': "top",
    })
    worksheet.set_column('A:E', 50, wrap_format)
    writer.save()


def main():
    pass


if __name__ == '__main__':
    main()