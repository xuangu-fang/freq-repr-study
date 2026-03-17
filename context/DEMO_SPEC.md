# Demo Specifications

## Demo A: phase_family

Definition:
u_w(x,y) = A(x,y) * cos(w * tau(x,y) + phi(x,y))

A(x,y):
- smooth random amplitude field

tau(x,y):
- smooth travel-time-like field

phi(x,y):
- smooth phase offset field

One trajectory:
- fix A, tau, phi
- sweep w over frequency grid

## Demo B: screened_poisson

PDE:
(Delta - alpha^2) u_alpha = s(x,y)

Domain:
- 2D periodic box
- solved by FFT

s(x,y):
- smooth random source field

One trajectory:
- fix s
- sweep alpha over parameter grid