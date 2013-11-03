// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuComplex.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMul(Complex *, int);

// Padding functions
int PadData(const Complex *, Complex **, int);

// FFtShift functions
void Allocate(int *pos_array, const int size, const int init);
Complex * FftShift(Complex *inSignal, const int height, const int width);

// Print functions
void PrintData(const Complex *, int , int, int);

// Assumes a square array.
#define WIDTH			   40
#define HEIGHT			   40
const int size1d = WIDTH * HEIGHT;
const int width_padded = 2 * WIDTH - 1;
const int height_padded = 2 * HEIGHT - 1;

int main(int argc, char **argv)
{
    printf("2D Auto-correlator is starting...\n");

    findCudaDevice(argc, (const char **)argv);

    // Allocate host memory for the signal
    Complex *h_signal = (Complex *)malloc(sizeof(Complex) * size1d);

	// Create an input array of all 5s.
    for (unsigned int i = 0; i < size1d; ++i)
    {
        h_signal[i].x = (float)5;
        h_signal[i].y = 0;
    }

    // Pad signal and filter kernel
    Complex *h_padded_signal;
    int new_size = PadData(h_signal, &h_padded_signal, size1d);
    int mem_size = sizeof(Complex) * new_size;

    // Allocate device memory for signal
    Complex *d_signal;
    checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));

    // CUFFT plan
    cufftHandle plan;
    checkCudaErrors(cufftPlan2d(&plan, width_padded, height_padded, CUFFT_C2C));

    // Transform signal and kernel
    printf("Transforming signal cufftExecC2C\n");
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));

    // Multiply the coefficients together and normalize the result
    printf("Launching ComplexPointwiseMul<<< >>>\n");
	dim3 threadsPerBlock(height_padded, 1);
	dim3 numBlocks(width_padded, 1);
	if (threadsPerBlock.x > 1024) {
		threadsPerBlock.x = 1024;
	}
	if (numBlocks.x > 1024) {
		numBlocks.x = 1024;
	}
    ComplexPointwiseMul<<<numBlocks, threadsPerBlock>>>(d_signal, new_size);

    // Check if kernel execution generated and error
    getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n\n");
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));

    // Copy device memory to host
    Complex *h_correllated_signal = h_padded_signal;
    checkCudaErrors(cudaMemcpy(h_correllated_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

	printf("Auto-correlated array...\n");
	// Shifting elements into appropriate positions
	h_correllated_signal = FftShift(h_correllated_signal, height_padded, width_padded);	
	PrintData(h_correllated_signal, height_padded, width_padded, new_size);

    // cleanup memory
    free(h_signal);
    free(h_padded_signal);
	free(h_correllated_signal);
    checkCudaErrors(cudaFree(d_signal));

    cudaDeviceReset();
	exit(true ? EXIT_SUCCESS : EXIT_FAILURE);
}

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size)
{
    int new_size = width_padded * height_padded;

	int index1d = 0;
	int index1d_padded = 0;

    // Pad signal
    Complex *new_data = (Complex *)malloc(sizeof(Complex) * new_size);
	int height = 0;
	while (height < height_padded && index1d < size1d) {
		for (int i = 0; i < WIDTH; ++i) {
			new_data[index1d_padded] = signal[index1d];
			index1d_padded++;
			index1d++;
		}
		for (int i = 0; i < width_padded - WIDTH; ++i) {
			new_data[index1d_padded].x = 0;
			new_data[index1d_padded].y = 0;
			index1d_padded++;
		}
		height++;
	}

	while (height < height_padded) {
		for (int i = 0; i < width_padded; ++i) {
			new_data[index1d_padded].x = 0;
			new_data[index1d_padded].y = 0;
			index1d_padded++;
		}
		height++;
	}

    *padded_signal = new_data;

    return new_size;
}

// Print the last 50 values of the array.
void PrintData(const Complex *data, int height, int width, int scale)
{
	int diff;
	if (width >= 50) {
		diff = 50;
	} else {
		diff = width;
	}

	for (unsigned int i = height-1; i < height; ++i)
    {
		for (unsigned int j = width-diff; j < width; ++j)
		{
			printf("%0.2f ", data[i*height+j].x/scale);
		}
		printf("\n");
    }
	printf("\n");
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMul(Complex *d_signal, int size)
{
	const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads) {
		d_signal[i] = ComplexMul(d_signal[i], cuConjf(d_signal[i]));
	}
}

// Allocates the positions of the correllated signal in the input arrays.
void Allocate(int *pos_array, const int size, const int init)
{
	// FFT shift algorithm works by starting at halfway of the width and height dimensions
	// and counting up to its max dimensions. Then adding the dimensions from the start to 
	// halfway.
	int j = 0;
	for(int i = init; i < size; ++i)
	{
		pos_array[j] = i;
		++j;
	}

	for(int i = 0; i < init; ++i)
	{
		pos_array[j] = i;
		++j;
	}
}

// Performs the Fftshift on the input correllated array.
Complex * FftShift(Complex *inSignal, const int height, const int width)
{
	// Halves the heights and widths rounded up.
	int sz[2] = {(height + 1) / 2, (width + 1) / 2};
	
	int *x_pos = (int *) malloc(width*sizeof(int));
	int *y_pos = (int *) malloc(height*sizeof(int));
	Complex *Buf = (Complex *)malloc(sizeof(Complex)*height*width);

	// Allocates the positions to access the correllated array to create the shifted array.
	Allocate(y_pos, height, sz[0]);
	Allocate(x_pos, width, sz[1]);

	// Uses the x_pos and y_pos arrays to access rearrange the correllated array.
	for(int i = 0; i < height; ++i)
	{
		for(int j = 0; j < width; ++j)
		{
			Buf[j + i*width] = inSignal[y_pos[i]*width + x_pos[j]];
		}
	}	
	free (x_pos);
	free (y_pos);	
	
	// Returns the new FftShifted array.
	return Buf;
}
