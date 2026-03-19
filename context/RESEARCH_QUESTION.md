# Research Questions

Q1. In which representation is the trajectory w -> u_w smoother?
Q2. In which representation is the trajectory lower-dimensional?
Q3. Which representation yields better interpolation of missing frequencies?
Q4. Are these conclusions consistent across demo families?

## Complex Field Specific Questions (for Helmholtz equation)

Q5. For complex-valued PDE solutions (Helmholtz equation), which representation better captures the frequency trajectory:
   a) Amplitude-phase representation (nonlinear, physical)
   b) Real-imaginary representation (linear, mathematical)

Q6. How does phase unwrapping affect the smoothness and interpolability of the frequency trajectory?
Q7. Does the amplitude-phase representation provide better physical interpretability compared to real-imaginary?
Q8. For complex fields, are there intrinsic differences in how amplitude and phase vary with frequency compared to real and imaginary parts?

Q9. Can learning-based representations (e.g., neural autoencoders) outperform traditional representations for complex field trajectories?
Q10. What are the trade-offs between interpretability (amplitude-phase) and performance (learning-based) for complex PDE solutions?

## Non-goals
- No diffusion model training yet
- No geometry
- # No Helmholtz scattering  # Temporarily allowed for representation generalization study
- No neural architecture search
- Note: Helmholtz and wave equations may be implemented for testing representation generalization across PDE types
- Note: Learning-based representations are now allowed for comparative study of representation quality