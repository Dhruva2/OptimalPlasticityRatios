# OptimalPlasticityRatios

[![Build Status](https://ci.appveyor.com/api/projects/status/github/Dhruva2/OptimalPlasticityRatios.jl?svg=true)](https://ci.appveyor.com/project/Dhruva2/OptimalPlasticityRatios-jl)


Code for the figures of the paper:
**Optimal plasticity for memory maintenance in the presence of synaptic fluctuations**

- The quality of the code isn't great. I made it when I was just learning Julia, and just packaged it up during final paper submission, so others can verify the results of the paper, rather than rewriting it entirely.

To run:

1. git clone the repository
2. Open julia 1.3 in the repository. **This package doesn't work with earlier or later versions of Julia**.
2. activate the environment with 
```
]activate .
using OptimalPlasticityRatios
``` 
3. run a script (e.g. linear figure) with
```
include("scripts/linear_figure.jl")
```.
