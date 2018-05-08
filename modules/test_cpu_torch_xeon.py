from common import PerfTestCase

tc = PerfTestCase()

tc.measure(test_name='torch.contiguousCopy',
        stmt='''
y = x.clone()
''',
        setup='''
import torch
x = torch.rand(3,1300,1200,10)
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

tc.measure(test_name='torch.contiguousMul',
        stmt='''
y = A * 5.5
''',
        setup='''
import torch
A = torch.randn(3, 1300, 1200, 10)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousLshift',
        stmt='''
y = A << 2
''',
        setup='''
import torch
A = torch.randn(3, 1300, 1200, 10)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousRshift',
        stmt='''
y = A >> 2
''',
        setup='''
import torch
A = torch.randn(3, 1300, 1200, 10)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousFMOD1',
        stmt='''
y = A.fmod(2)
''',
        setup='''
import torch
L = list(range(0,5*1000*1000))
A = torch.tensor(L)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousFMOD2',
        stmt='''
y = A % 2
''',
        setup='''
import torch
L = list(range(0,100*1000*1000))
A = torch.tensor(L)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousRemainder',
        stmt='''
y = A.remainder(5.5)
''',
        setup='''
import torch
L = list(range(0,100*1000*1000))
A = torch.tensor(L)
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

tc.measure(test_name='torch.contiguousBitAnd',
        stmt='''
y = A & 5
''',
        setup='''
import torch
L = list(range(0, 100*1000*1000))
A = torch.tensor(L, dtype=torch.int64)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousBitOr',
        stmt='''
y = A | 5
''',
        setup='''
import torch
L = list(range(0, 100*1000*1000))
A = torch.tensor(L, dtype=torch.int64)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousBitXor',
        stmt='''
y = A ^ 5
''',
        setup='''
import torch
L = list(range(0, 100*1000*1000))
A = torch.tensor(L, dtype=torch.int64)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousClamp',
        stmt='''
y = A.clamp(100, 500)
''',
        setup='''
import torch
L = list(range(0, 100*1000*1000))
A = torch.tensor(L, dtype=torch.int64)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousPow',
        stmt='''
y = A.pow(2)
''',
        setup='''
import torch
A = torch.randn([3, 3000, 3000, 10],dtype=torch.float)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousLog',
        stmt='''
y = A.log()
''',
        setup='''
import torch
A = torch.randn([3, 2000, 2000, 10])
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousLGAMMA',
        stmt='''
y = A.lgamma()
''',
        setup='''
import torch
A = torch.randn([3, 100, 100, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousDiGAMMA',
        stmt='''
y = A.digamma()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousLog1p',
        stmt='''
y = A.log1p()
''',
        setup='''
import torch
A = torch.randn([3, 500, 500, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousSigmoid',
        stmt='''
y = A.sigmoid()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousExpm1',
        stmt='''
y = A.expm1()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousCos',
        stmt='''
y = A.cos()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousAcos',
        stmt='''
y = A.acos()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousCosh',
        stmt='''
y = A.cosh()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousAsin',
        stmt='''
y = A.asin()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousSinh',
        stmt='''
y = A.sinh()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousTan',
        stmt='''
y = A.tan()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousAtan',
        stmt='''
y = A.atan()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousTanh',
        stmt='''
y = A.tanh()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousErf',
        stmt='''
y = A.erf()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousErfinv',
        stmt='''
y = A.erfinv()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousSqrt',
        stmt='''
y = A.sqrt()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 8], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousRsqrt',
        stmt='''
y = A.rsqrt()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 8], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousCeil',
        stmt='''
y = A.ceil()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousFloor',
        stmt='''
y = A.floor()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousRound',
        stmt='''
y = A.round()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousAbs',
        stmt='''
y = A.abs()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousTrunc',
        stmt='''
y = A.trunc()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousFrac',
        stmt='''
y = A.frac()
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousCinv',
        stmt='''
y = 1.0 / A
''',
        setup='''
import torch
A = torch.randn([3, 300, 300, 10], dtype=torch.double)
''',
        number=500,
        repeat=20)

tc.measure(test_name='torch.contiguousConv2D',
        stmt='''
output = net(data)
''',
        setup='''
import torch
import torch.nn as nn
from torch_nn_models import Conv2D
data = torch.randn(1,3,512,512)
net = Conv2D()
''',
        number=500,
        repeat=20)

