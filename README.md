# CUDA Multigrid
The code in this repository solves Poisson's equation in 2D subject to Dirichlet boundary condition
using the Multigrid method with a Gauss-Seidel smoother. The PDE is discretized using second order
finite differences and conservative prolongation and restriction operators. There are two
implementations of the solver, a single-threaded CPU solver and a GPU solver (custom CUDA
kernels). To expose parallelism, the GPU solver uses a Red-Black Gauss-Seidel smoother. 

## Usage
Look at the test program `test/test_poisson.cu` to learn how to use the code.

```CUDA
// Select problem to solve
using CUDAProblem = CUDAPoisson<L1NORM, Number>;
// Select smoother
using CUDASmoother = CUDAGaussSeidelRedBlack;
// Select multigrid solver
using CUDAMG = CUDAMultigrid<CUDASmoother, CUDAProblem, Number>;

CUDAProblem problem(l, h, modes);
CUDAMG solver(problem);
auto out = solve(solver, problem, opts);
printf("Iterations: %d, Residual: %g \n", out.iterations, out.residual);
```


```
CPU: Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
GPU: NVIDIA RTX 2080 Ti

test/test_poisson 
Solver: Gauss-Seidel (red-black) 
Iteration 	 Residual
10      	 6.674289 
20      	 1.369973 
30      	 0.2812023 
40      	 0.05771992 
50      	 0.01184766 
60      	 0.002431866 
70      	 0.0004991677 
80      	 0.0001024598 
90      	 2.103102e-05 
100     	 4.316853e-06 
110     	 8.860825e-07 
120     	 1.818784e-07 
130     	 3.733259e-08 
Iterations: 139, Residual: 8.97769e-09 
Solver: Multi-Grid<Gauss-Seidel (red-black)> 
Iteration 	 Residual
10      	 4.959894e-09 
Iterations: 10, Residual: 4.95989e-09 
Solver: CUDA Gauss-Seidel (red-black) 
Iteration 	 Residual
10      	 6.674289 
20      	 1.369973 
30      	 0.2812023 
40      	 0.05771992 
50      	 0.01184766 
60      	 0.002431866 
70      	 0.0004991677 
80      	 0.0001024598 
90      	 2.103102e-05 
100     	 4.316853e-06 
110     	 8.860825e-07 
120     	 1.818784e-07 
130     	 3.733259e-08 
Iterations: 139, Residual: 8.97769e-09 
Solver: CUDA Multi-Grid<CUDA Gauss-Seidel (red-black)> 
Iteration 	 Residual
10      	 4.959894e-09 
Iterations: 10, Residual: 4.95989e-09 
MMS convergence test
Solver: Multi-Grid<Gauss-Seidel (red-black)> 
Grid Size 	 Iterations 	 Time (ms) 	 Residual 	 Error 		 Rate 
   6 x 6    	 1       	 0.00406 	 2.4718e-16 	 0.9348 	 -inf  
  10 x 10   	 8       	 0.01162 	 8.3216e-09 	 0.30908 	 1.59669 
  18 x 18   	 10      	 0.03814 	 4.9599e-09 	 0.08183 	 1.91727 
  34 x 34   	 11      	 0.14534 	 1.6851e-09 	 0.02074 	 1.98024 
  66 x 66   	 11      	 0.55350 	 2.2692e-09 	 0.0052025 	 1.99512 
 130 x 130  	 11      	 2.19837 	 2.4601e-09 	 0.0013017 	 1.99878 
 258 x 258  	 11      	 8.77101 	 2.5173e-09 	 0.0003255 	 1.99970 
 514 x 514  	 11      	 40.58800 	 2.5417e-09 	 8.1378e-05 	 1.99993 
1026 x 1026 	 11      	 179.36189 	 2.5836e-09 	 2.0344e-05 	 2.00001 
2050 x 2050 	 11      	 746.79437 	 2.735e-09 	 5.0858e-06 	 2.00010 
4098 x 4098 	 11      	 3020.28027 	 3.38e-09 	 1.2711e-06 	 2.00041 
8194 x 8194 	 11      	 12569.50879 	 7.1411e-09 	 3.174e-07 	 2.00166 
MMS convergence test
Solver: CUDA Multi-Grid<CUDA Gauss-Seidel (red-black)> 
Grid Size 	 Iterations 	 Time (ms) 	 Residual 	 Error 		 Rate 
   6 x 6    	 1       	 0.08147 	 2.4718e-16 	 0.9348 	 -inf  
  10 x 10   	 8       	 0.46346 	 8.3216e-09 	 0.30908 	 1.59669 
  18 x 18   	 10      	 0.76461 	 4.9599e-09 	 0.08183 	 1.91727 
  34 x 34   	 11      	 1.04093 	 1.6851e-09 	 0.02074 	 1.98024 
  66 x 66   	 11      	 1.23590 	 2.2692e-09 	 0.0052025 	 1.99512 
 130 x 130  	 11      	 1.52275 	 2.4601e-09 	 0.0013017 	 1.99878 
 258 x 258  	 11      	 2.05760 	 2.5173e-09 	 0.0003255 	 1.99970 
 514 x 514  	 11      	 3.58195 	 2.5417e-09 	 8.1378e-05 	 1.99993 
1026 x 1026 	 11      	 10.08339 	 2.5836e-09 	 2.0344e-05 	 2.00001 
2050 x 2050 	 11      	 29.20931 	 2.735e-09 	 5.0858e-06 	 2.00010 
4098 x 4098 	 11      	 109.10480 	 3.38e-09 	 1.2711e-06 	 2.00041 
8194 x 8194 	 11      	 397.84283 	 7.1411e-09 	 3.174e-07 	 2.00166 
```

## TODO
Currently, the GPU solver is more or less a copy-and-paste-version of the CPU solver. It would be
nice to remove much of the code duplication. There's also room for improvement in terms of
optimization. First, I should perform a baseline profiling to identify the current hotspots.
Nonetheless, I already have some ideas on what can be improved. The red-black Gauss-Seidel
implementation is currently terrible. It takes two trips to DRAM to load the data. First, once for
the red points, and then once more for the black points. At each time, only half of the threads are
active. It should possible to keep all threads active by loading the data into shared memory
using a float2/double2 instruction. Some kernels like the restriction and residual kernel can also
probably merged into a single one to further reduce memory traffic. 
