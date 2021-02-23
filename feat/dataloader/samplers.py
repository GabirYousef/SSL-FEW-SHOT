import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        # print("m_ind: ", len(self.m_ind))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            # print("i_batch: ", i_batch)
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            # print("\nclasses: ", classes)
            for c in classes:
                l = self.m_ind[c]
                # print("\nl: ", l)
                pos = torch.randperm(len(l))[:self.n_per]
                # print("\npos: ", pos)
                batch.append(l[pos])
                # print("\n\nlen of l[pos]: ", l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # print("batch data: ", batch)
            # print("batch size: ", batch.size())
            yield batch

