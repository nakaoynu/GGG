import numpy as np
from circular_basis_transmission_spectrum import get_stevens_operators

spin = 7 / 2  # Gd3+のスピン量子数
m = np.arange(spin, -spin - 1, -1)
print("m values:", m)

O04, O44, O06, O46 = get_stevens_operators()
print("Stevens operators:")
print("O04:", O04)
print("O44:", O44)
print("O06:", O06)
print("O46:", O46)
