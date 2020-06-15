
from hw2_functions import solve_heat_rod

dts = [0.0001, 0.001, 0.01]

solve_heat_rod(alpha=1, dt=0.001)
for i in dts:
    solve_heat_rod(alpha=1, dt=i)

for i in dts:
    solve_heat_rod(alpha=0.5, dt=i)
solve_heat_rod(alpha=0.5, dt=0.02)

for i in dts:
    solve_heat_rod(alpha=0, dt=i)
solve_heat_rod(alpha=0, dt=0.025)

