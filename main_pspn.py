#!/usr/bin/env python3

from rich.traceback import install
import torch

install(suppress=[torch])
from simple_einet.distributions.normal import Normal
from simple_einet.einet import PSPN, EinetConfig, EinetColumn, EinetColumnConfig

from icecream import install

install()


def train(pspn: PSPN):
    optim = torch.optim.Adam(params=pspn.columns[-1].parameters())

    for i in range(5):
        optim.zero_grad()
        x = torch.randn(5, 16)
        lls = pspn(x)  # p(x | y)
        nlls = -1 * lls.sum()
        nlls.backward()
        optim.step()


def main():
    config = EinetColumnConfig(
        num_channels=1,
        num_features=16,
        num_sums=5,
        num_leaves=5,
        num_repetitions=1,
        depth=3,
        leaf_type=Normal,
        leaf_kwargs={},
        num_classes=2,
        seed=0,
    )

    pspn = PSPN(config)

    num_tasks = 3
    for task_index in range(num_tasks):
        ic(task_index)
        pspn.expand()
        train(pspn)


if __name__ == "__main__":
    main()
