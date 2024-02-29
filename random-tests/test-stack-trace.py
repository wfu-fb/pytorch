import torch
import torch.nn as nn
from torch.export import export, Dim
from functorch.experimental.control_flow import cond

class Apple(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
    def forward(self, x):
        return self.linear(x)

class Bar(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.lin = nn.Linear(4, 4)
        self.ap = Apple()
        self.buf = torch.randn(4)
    def forward(self, x):
        x = 2 * self.bn(self.lin(x))
        x += self.ap(x)
        return x

class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.bar = Bar()
    def forward(self, x):
        y = cond(x.sum() > 0, lambda x: x.sin(), lambda x: x.cos(), [x])
        return self.bar(x) + self.bar.buf * y

if __name__ == '__main__':
    m = Foo()
    inputs = (torch.randn(4, 4),)
    ep = export(m, inputs, strict=False)

    for node in ep.graph.nodes:
        print(node, node.op)
        # print("source_fn_stack:", node.meta.get('source_fn_stack', None))
        print("stack_trace:", node.meta.get('stack_trace', None), type(node.meta.get('stack_trace', None)))
        print()
