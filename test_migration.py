import _migration
import numpy as np
import random

a=[random.choices(range(10,80), k=5) for _ in range(6)]
res=_migration.array_buffer(a)
print(res)