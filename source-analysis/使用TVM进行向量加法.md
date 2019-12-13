
```python
import tvm
import numpy as np

m = 3
ctx = tvm.context("cpu", 1)
A = tvm.placeholder((m,), name='A')
B = tvm.compute((m,), lambda i: A[i]+2, name='B')
s = tvm.create_schedule(B.op)
fm = tvm.build(s, [A, B], "cpu", target_host="llvm", name="mul")

ff = fm.get_function("mul")
ff = fm.entry_func

a = tvm.nd.array(np.zeros(m, A.dtype), ctx)
b = tvm.nd.array(np.zeros(m, A.dtype), ctx)
res = ff(a, b)
print(b)
```
输出结果：
[2. 2. 2.]
