# RMSD Computation tool
This tool computes the root-mean-square deviation 
between two sets of points. This is useful for comparing 
the similarity of atomic structres, and is used to find
homologous protein structures

### The code
I have written a small class to read in my fake data. The 
format is just three integers per line. I plan to add `.pdb` 
reading capability soon. This class returns a stl vector
containng a bunch of point struct pointers. These objects can
be passed to the actual RMSD functions.

There is both a CPU and GPU version of the RMSD function. The 
CPU one is faster, CUDA doesnt provide a reduce operation so
it's not really a great choice for this. I wanted to practice 
writing GPU code anyway.

In addition to adding `.pdb` support, I want to create a way to
minimize the RMSD between two structures. That may actually benefit 
from a GPU port.