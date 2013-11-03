parallel-2D-autocorrelator
==========================

A parallel program to compute the auto-correlation of a 2D input array using the CUDA platform. It was written as part of a group project during the final year of my engineering degree at the University of Canterbury.

From the report:

"A parallel program has been written, using the CUDA platform, to perform 2D autocorellation. The auto-correlation algorithm is based on the Wiener-Khinchin theorem and works as follows. Firstly the input array is padded, and then data is copied to the Graphics Processing Unit (GPU). The GPU performs a forward Fast Fourier Transform (FFT), multiplication of each value in the array by its complex conjugate and an inverse FFT. Finally, the results are copied back to the CPU and the shifted results printed to the screen. Performance measurements show the program takes 0.63 seconds to correlate a square input array with dimensions of 512. Our program was found to outperform an implementation run on the CPU for square arrays with dimensions over 1500. At a size of 4000, the CUDA program was more than 11 times faster."