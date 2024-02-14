This is a Matlab implementation of the dictionary learning algorithm proposed in
S. Hawe, M. Seibert, and M. Kleinsteuber. Separable Dictionary Learning. 
In IEEE Conference on Computer Vision and Pattern Recognition, pp. 1--8. 2013.

This implementation requires the open source MATLAB Tensor Toolbox Version 2.5, which can be 
downloaded from http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html (in file main.m
you can see that I store this toolbox in the same folder as the SeDiL folder and add it 
to the matlab path via addpath('../tensor_toolbox_2.5/'); Feel free to change this.)

Once you have downloaded and installed the tensor toolbox, you can learn a dictionary by 
executing the main.m script. In the given example, a separable dictionary with dimension 
D1 = 16 x 8 and D2 = 16 x 8 is learned from 8 x 8 patches. As an example of how to learn 
a normal non-separable dictionary, just set no_sep = 1 in line 23 of the main.m file. The
resulting dictionary is of dimension 256 x 64