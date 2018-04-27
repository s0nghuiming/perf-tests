from common import PerfTestCase

tc = PerfTestCase()

tc.measure(test_name='torch.contiguousCopy',
        stmt='''
y = x.clone()
''',
        setup='''
import torch
x = torch.rand(3,1300,120,100)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousAdd',
        stmt='''
y = A + 5.5
''',
        setup='''
import torch
A = torch.randn(3, 1300, 120, 100)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousDiv',
        stmt='''
y = torch.div(A, 3)
''',
        setup='''
import torch
A = torch.randn(3, 1300, 120, 100)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousSin',
        stmt='''
y = torch.sin(A)
''',
        setup='''
import torch
A = torch.randn(3, 1300, 120, 100)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousExp',
        stmt='''
y = torch.exp(A)
''',
        setup='''
import torch
A = torch.randn(3, 1300, 120, 100)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousSum',
        stmt='''
y = torch.sum(A, 0)
''',
        setup='''
import torch
A = torch.randn(3, 1300, 120, 100)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousProd',
        stmt='''
y = torch.prod(A, 0)
''',
        setup='''
import torch
A = torch.randn(3, 1300, 120, 100)
''',
        number=500,
        repeat=20)

