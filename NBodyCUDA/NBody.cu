#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USER_NAME "elt18sx"		//replace with your username

#define FILE_CACHE_SIZE 255		//cache used to store one line content while reading file

struct argument {
	// Arguments to store essential parameters, such
	// as N, D, M and I, etc.

	unsigned int n;
	unsigned int d;
	enum MODE m;
	unsigned int iter;
	char* input_file;
	boolean visualisation;
};

struct point {
	// An Structure to store float data in both x and y
	// axis. Usually, it is used to store the acceleration

	float x;
	float y;
};

void print_help();
void step(void);

void raise_error(char* error_message, boolean print_msg, int exit_type);
struct argument load_args(int argc, char* argv[]);
void read_file(struct argument args, struct nbody* bodies);
void generate_data(struct argument args, struct nbody* bodies);

struct point calculate_single_body_acceleration(struct nbody* bodies, int body_index, struct argument args);
void compute_volocity(struct nbody* bodies, float time_step, struct argument args);
void update_location(struct nbody* bodies, float time_step, struct argument args);
void update_heat_map(float* heat_map, struct nbody* bodies, struct argument args);

void checkCUDAErrors(const char* msg);

struct argument args;
struct nbody* h_bodies;
struct nbody* d_bodies;
float* heat_map;

__global__ void test(struct nbody* d_bodies){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("");
}

__device__ struct point calculate_single_body_acceleration1(struct nbody* bodies, int N) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	struct point acceleration = { 0,0 };
	struct nbody* target_bodies = bodies + index;
	for (unsigned int i = 0; i < N; i++) {
		struct nbody* external_body = bodies + i;
		if (i != index) {
			float x_diff = external_body->x - target_bodies->x;
			float y_diff = external_body->y - target_bodies->y;
			//float r = sqrt((double)x_diff * x_diff + (double)y_diff * y_diff);
			double r = (double)x_diff * x_diff + (double)y_diff * y_diff;
			float temp = G * external_body->m / (float)(sqrt((r + SOFTENING * SOFTENING)) * (r + SOFTENING * SOFTENING));
			//float temp = G_const * external_body->m / (float)pow(((double)r + SOFTENING_square), 3.0 / 2);
			acceleration.x += temp * x_diff;
			acceleration.y += temp * y_diff;
			printf("");
			}
		}
	return acceleration;
}

__global__ void compute_volocity1(struct nbody* bodies) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	struct point acceleration = calculate_single_body_acceleration1(bodies,1024);
	struct nbody* target_bodies = bodies + index;
	target_bodies->vx += acceleration.x * dt;
	target_bodies->vy += acceleration.y * dt;
}

__global__ void update_location1(struct nbody* bodies) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	struct nbody* target_bodies = bodies + index;
	target_bodies->x += target_bodies->vx * dt;
	target_bodies->y += target_bodies->vy * dt;
}


int main(int argc, char* argv[]) {
	//Processes the command line arguments
		//argc in the count of the command arguments
		//argv is an array (of length argc) of the arguments. The first argument is always the executable name (including path)
	args = load_args(argc, argv);
	if (args.m == OPENMP)
		omp_set_num_threads(omp_get_max_threads());
	//omp_set_num_threads(10);

	//Allocate any heap memory
	int size = sizeof(struct nbody) * args.n;
	h_bodies = (struct nbody*) malloc(size);
	heat_map = (float*)malloc(sizeof(float) * args.d * args.d);

	//Depending on program arguments, either read initial data from file or generate random data.
	if (args.input_file != NULL)
		read_file(args, h_bodies);
	else
		generate_data(args, h_bodies);

	//Allocate CUDA memory & copy data
	if (args.m == CUDA) {
		cudaMalloc((void**)&d_bodies, size);
		cudaMemcpy(d_bodies, h_bodies, size, cudaMemcpyHostToDevice);
		checkCUDAErrors("Input transfer to device");

		//dim3 blocksPerGrid(8, 1, 1);
		//dim3 threadsPerBlock(128, 1, 1);

		//test << < blocksPerGrid, threadsPerBlock >> > (d_bodies);
		//compute_volocity1 << < blocksPerGrid, threadsPerBlock >> > (d_bodies);
		//cudaThreadSynchronize();
		//update_location1 << < blocksPerGrid, threadsPerBlock >> > (d_bodies);
		//cudaThreadSynchronize();
	}


	//Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	//args.visualisation = TRUE;
	if (args.visualisation == TRUE) {
		initViewer(args.n, args.d, args.m, &step);
		setNBodyPositions(h_bodies);
		setHistogramData(heat_map);
		startVisualisationLoop();
	}
	else {
		clock_t tic = clock();
		//Sleep(123456);
		step();
		clock_t toc = clock();
		int seconds = (toc - tic) / CLOCKS_PER_SEC;
		int milliseconds = (toc - tic - seconds * CLOCKS_PER_SEC);
		printf("Execution time %d seconds %d milliseconds\n", seconds, milliseconds);
	}

	free(h_bodies);
	free(heat_map);
	return 0;
}

void step(void)
{
	/*------------------------------------------------------
	A single step to update all bodies.
	1. compute the volocity for all bodies.
	2. update the location for all bodies accoreding to their
	present speed(volocity).

	Args:
		args: A structure which is used to store a set of
				parameters.
		time_step: The time refers to dt.

	Return:
		void
	--------------------------------------------------------*/

	float time_step = dt;
	for (unsigned int i = 0; i < args.iter; i++) {
		compute_volocity(h_bodies, time_step, args);
		#pragma omp barrier
		if (args.m == CUDA)
			cudaThreadSynchronize();
		update_location(h_bodies, time_step, args);
		if (args.m == CUDA)
			cudaThreadSynchronize();
		if (args.visualisation == TRUE)
			update_heat_map(heat_map, h_bodies, args);
	}
}

void print_help() {
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}

void raise_error(char* error_message, boolean print_msg, int exit_type) {
	/*------------------------------------------------------
	When encounters an error, the scheduled output will be
	displayed and the program will exit.

	For example,
		`raise_error("Error:...",TRUE,1)` will first print
		the error message "Error:...", and then exit with
		exit code 1.

	Args:
		error_message: Message to output
		print_msg: A flag of whether to output the message
		exit_type: Exit type, for example, 1, 2 or 3

	Return:
		void
	------------------------------------------------------*/

	printf("%s\n", error_message);
	if (print_msg)
		print_help();
	if (exit_type != 0)
		exit(exit_type);
}

struct argument load_args(int argc, char* argv[]) {
	/*------------------------------------------------------
	Check the validity of input parameters from command line

	For example,
		with the arguments of "1024 50 OPENMP -i 1000", all
		the arguments will be stored in an 'argument' structure.
		In the current example, the data will looks like,
			argc.n: 1024
			argc.d: 50
			argc.m: OPENMP
			argc.iter: 1000
			argc.input_file: NULL
			argc.visualisation: FALSE

	Args:
		argc: The number of input parameters, and should be
			always equal or larger than 1.
		argv: An array which stores the parameters as string.
			In addition, `argv[0]` is always the PATH of the
			program.
		args: A structure which is used to store a set of
			parameters.

	Return:
		args: An argument struct that stores the arguments.

	Raises:
		exit(1): encounter with unexpected input.
	--------------------------------------------------------*/

	struct argument args = { 200,8,CPU,1,NULL,TRUE };

	// Part 1: check the number of argc
	if (argc == 1)
		return args;
	else if (argc != 4 && argc != 6 && argc != 8)
		raise_error("Eorror: Incomplete parameters.", TRUE, 1);
	else {
		// Part 2: check arguments N ,D and Mode validation
		if (!atoi(argv[1]) || !atoi(argv[2]))
			raise_error("Error: N and D should be a number.", TRUE, 1);
		else {
			args.n = atoi(argv[1]); // N
			args.d = atoi(argv[2]); // D
		}
		if (strcmp(argv[3], "CUDA") != 0 && strcmp(argv[3], "CPU") != 0 && strcmp(argv[3], "OPENMP") != 0)
			raise_error("Error: mode should be CPU or OPENMP", TRUE, 1);
		else {
			if (strcmp(argv[3], "CPU") == 0)
				args.m = CPU;
			else if (strcmp(argv[3], "OPENMP") == 0)
				args.m = OPENMP;
			else
				args.m = CUDA;
		}


		// Part 3: check options
		if (argc == 6 || argc == 8) {
			if (strcmp(argv[4], "-i") == 0) {
				args.iter = atoi(argv[5]);
				args.visualisation = FALSE;
				if (argc == 8 && strcmp(argv[6], "-f") == 0)
					args.input_file = argv[7];
			}
			else if (strcmp(argv[4], "-f") == 0) {
				args.input_file = argv[5];
				if (argc == 8 && strcmp(argv[6], "-i") == 0) {
					args.iter = atoi(argv[7]);
					args.visualisation = FALSE;
				}
			}
		}
	}
	return args;
}

void read_file(struct argument args, struct nbody* bodies) {
	/*------------------------------------------------------
	Read the data from file according to the arguments

	Args:
		args: A structure which is used to store a set of
			parameters.
		f: A pointer to read file.
		buff: An array work as cache to store a line of content
		file_index: The index of data line, and comments are
			already ignored.
		token, tokenremain, delims: char pointer used for
			separate function.
		body_member: A single body data.

	Return:
		void

	Raises:
		exit(1): if file doesn't exist.
		exit(1): if the number of data in file is not equal
			to N.
	--------------------------------------------------------*/

	FILE* f = fopen(args.input_file, "r");
	if (f == NULL)
		raise_error("Error: file doesn't exist.", FALSE, 1);
	char buff[FILE_CACHE_SIZE];
	int file_index = 0;
	while (fgets(buff, FILE_CACHE_SIZE, (FILE*)f)) //lines
		if (buff[0] != '#') {
			//printf("%s", buff);
			char* token;
			char delims[] = ",";
			char* tokenremain = buff;
			float* body_member = (float*)&bodies[file_index++];
			int i = 0;

			for (token = strtok(buff, delims); i < 5; token = strtok(NULL, delims)) {
				//printf("%s\n", token);
				if (token != NULL && strlen(token) > 1)
					*(body_member + i++) = atof(token);
				else {
					if (i == 0 || i == 1) *(body_member + i++) = (float)rand() / 0x8000;
					else if (i == 2 || i == 3) *(body_member + i++) = 0.0;
					else *(body_member + i++) = 1.0 / args.n;
				}
			}
		}
	fclose(f);
	if (file_index != args.n)
		raise_error("Error: the data in file is incompatible with argument N.", FALSE, 1);
}

void generate_data(struct argument args, struct nbody* bodies) {
	/*------------------------------------------------------
	Generate random data for all bodies.

	Args:
		args: A structure which is used to store a set of
				parameters.
		file_index: The index of data line.
		body_member: A single body data.

	Return:
		void
	--------------------------------------------------------*/

	for (unsigned int file_index = 0; file_index < args.n;) {
		float* body_member = (float*)&bodies[file_index++];
		for (int i = 0; i < 5;) {
			if (i == 0 || i == 1) *(body_member + i++) = (float)rand() / 0x8000;
			else if (i == 2 || i == 3) *(body_member + i++) = 0.0;
			else *(body_member + i++) = 1.0 / args.n;
		}
	}
}

struct point calculate_single_body_acceleration(struct nbody* bodies, int body_index, struct argument args) {
	/*------------------------------------------------------
	For a single body, calculate the total acceleration
	generated from other bodies.

	Args:
		args: A structure which is used to store a set of
				parameters.
		target_bodies: A single body data.
		SOFTENING_square: pre-calculate the square of SOFTING
			to accelerate the program speed.

	Return:
		acceleration: An structure with only 2 members which
			stores the accelerate in both x and y axis.
			For example,
				acceleration.x: 1.024
				acceleration.y: 2.048
	--------------------------------------------------------*/

	const float G_const = G;
	double SOFTENING_square = (double)SOFTENING * SOFTENING;
	struct point acceleration = { 0,0 };
	struct nbody* target_bodies = bodies + body_index;
	for (unsigned int i = 0; i < args.n; i++) {
		struct nbody* external_body = bodies + i;
		if (i != body_index) {
			float x_diff = external_body->x - target_bodies->x;
			float y_diff = external_body->y - target_bodies->y;
			//float r = sqrt((double)x_diff * x_diff + (double)y_diff * y_diff);
			double r = (double)x_diff * x_diff + (double)y_diff * y_diff;
			float temp = G_const * external_body->m / (float)(sqrt((r + SOFTENING_square)) * (r + SOFTENING_square));
			//float temp = G_const * external_body->m / (float)pow(((double)r + SOFTENING_square), 3.0 / 2);
			acceleration.x += temp * x_diff;
			acceleration.y += temp * y_diff;
		}
	}
	return acceleration;
}

void compute_volocity(struct nbody* bodies, float time_step, struct argument args) {
	/*------------------------------------------------------
	Calculate the volocity for bodies according to their
	accelerate.

	Args:
		args: A structure which is used to store a set of
				parameters.
		time_step: The time refers to dt.
		acceleration: An structure with only 2 members which
			stores the accelerate in both x and y axis.

	Return:
		void
	--------------------------------------------------------*/

	//double tic = omp_get_wtime();
	if (args.m == CPU) {
		for (unsigned int i = 0; i < args.n; i++) {
			struct point acceleration = calculate_single_body_acceleration(bodies, i, args);
			(bodies + i)->vx += acceleration.x * time_step;
			(bodies + i)->vy += acceleration.y * time_step;
		}
	}
	else if (args.m == OPENMP) {
		//omp_set_nested(1);
		int i;
#pragma omp parallel for schedule(dynamic,2)
		for (i = 0; i < args.n; i++) {
			struct point acceleration = calculate_single_body_acceleration(bodies, i, args);
			(bodies + i)->vx += acceleration.x * time_step;
			(bodies + i)->vy += acceleration.y * time_step;
		}
	}
	else if (args.m == CUDA) {
		dim3 blocksPerGrid(8, 1, 1);
		dim3 threadsPerBlock(128, 1, 1);
		compute_volocity1 << < blocksPerGrid, threadsPerBlock >> > (d_bodies);
	}
	//double t = omp_get_wtime() - tic;
	//printf("");
}

void update_location(struct nbody* bodies, float time_step, struct argument args) {
	/*------------------------------------------------------
	Calculate the new location for bodies according to their
	present location and speed.

	Args:
		args: A structure which is used to store a set of
				parameters.
		time_step: The time refers to dt.

	Return:
		void
	--------------------------------------------------------*/

	//double tic = omp_get_wtime();
	if (args.m == CPU) {
		for (unsigned int i = 0; i < args.n; i++) {
			(bodies + i)->x += (bodies + i)->vx * time_step;
			(bodies + i)->y += (bodies + i)->vy * time_step;
		}
	}
	else if (args.m == OPENMP) {
		int i;
#pragma omp parallel for schedule(dynamic,2)
		for (i = 0; i < args.n; i++) {
			(bodies + i)->x += (bodies + i)->vx * time_step;
			(bodies + i)->y += (bodies + i)->vy * time_step;
		}
	}
	else if (args.m == CUDA) {
		dim3 blocksPerGrid(8, 1, 1);
		dim3 threadsPerBlock(128, 1, 1);
		update_location1 << < blocksPerGrid, threadsPerBlock >> > (d_bodies);
	}
	//double t = omp_get_wtime() - tic;
	//printf("");
}

void update_heat_map(float* heat_map, struct nbody* bodies, struct argument args) {
	/*------------------------------------------------------
	Calculate the heat map based on present bodies.

	Args:
		args: A structure which is used to store a set of
				parameters.

	Return:
		void
	--------------------------------------------------------*/

	float grid_length = 1.0 / args.d;
	if (args.m == CPU) {
		// Initial heat map
		for (unsigned int i = 0; i < args.d * args.d; i++)
			*(heat_map + i) = 0.0;
		// Iterate over all data points
		for (unsigned int i = 0; i < args.n; i++) {
			struct point body_location = { (bodies + i)->x, (bodies + i)->y };
			if (!(body_location.x < 0 || body_location.x>1 || body_location.y < 0 || body_location.y>1)) {
				int row = (int)(body_location.x / grid_length);
				int line = (int)(body_location.y / grid_length);
				*(heat_map + line * args.d + row) += 1.0;
			}
		}
		// Normalize heat
		for (unsigned int i = 0; i < args.d * args.d; i++)
			*(heat_map + i) = *(heat_map + i) / args.n * args.d;
	}
	else if (args.m == OPENMP) {
		// Initial heat map
		int i;
#pragma omp parallel for schedule(dynamic,2)
		for (i = 0; i < args.d * args.d; i++)
			*(heat_map + i) = 0.0;
		// Iterate over all data points
#pragma omp parallel for
		for (i = 0; i < args.n; i++) {
			struct point body_location = { (bodies + i)->x, (bodies + i)->y };
			if (!(body_location.x < 0 || body_location.x>1 || body_location.y < 0 || body_location.y>1)) {
				int row = (int)(body_location.x / grid_length);
				int line = (int)(body_location.y / grid_length);
				//#pragma omp atomic
#pragma omp critical
				{*(heat_map + line * args.d + row) += 1.0; }
			}
		}
		// Normalize heat
#pragma omp parallel for schedule(dynamic,2)
		for (i = 0; i < args.d * args.d; i++)
			*(heat_map + i) = *(heat_map + i) / args.n * args.d;
	}
}

void checkCUDAErrors(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}