#include "CollisionDetection.cuh"

//kernels
__device__ void dSum(unsigned int* values, unsigned int* out) {
	__syncthreads();

	unsigned int threads = blockDim.x;
	unsigned int half = threads / 2;

	while (half) {
		if (threadIdx.x < half) {
			for (int k = threadIdx.x + half; k < threads; k += half) {
				values[threadIdx.x] += values[k];
			}
			threads = half;
		}
		half /= 2;
		__syncthreads();
	}

	if (!threadIdx.x) {
		atomicAdd(out, values[0]);
	}
}

__device__ void d_prefix_sum(unsigned int* values, unsigned int n) {
	int offset = 1;
	int a;
	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (threadIdx.x < d) {
			a = offset * (2 * threadIdx.x + 1) - 1;
			values[a + offset] += values[a];
		}
		offset <<= 1;
	}

	if (!threadIdx.x) {
		values[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;//128, 64, 32, 16, 8, 4, 2, 1
		__syncthreads();
		if (threadIdx.x < d) {//1, 2, 4, 8, 16, 32, 64, 128
			int a = offset * (2 * threadIdx.x + 1) - 1;
			float t = values[a];
			values[a] = values[a + offset];
			values[a + offset] += t;
		}
	}
}

__global__ void initData(Particle* particles, unsigned int* cells, unsigned int* objects, unsigned int* cell_count, ConstantsInitData constants) {
	extern __shared__ unsigned int s[];
	unsigned int count = 0;
	unsigned int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	float dist;
	float pos_x, pos_y;
	unsigned int cell_x;
	unsigned int cell_y;
	unsigned int l_id;
	int addValX, addValY;
	int tmp_cell;
	while (g_id < constants.objects_size) {
		count++;
		pos_x = particles[g_id].position.x;
		pos_y = particles[g_id].position.y;
		cell_x = (unsigned int)(pos_x / constants.diameter);
		cell_y = (unsigned int)(pos_y / constants.diameter);
		cells[4 * g_id] = (((cell_x << constants.x_shift) | cell_y) << 1) | 0x00;
		objects[4 * g_id] = g_id << 1 | 0x01;

		l_id = 1;

		addValX = -1;
		addValY = -1;

		tmp_cell = (unsigned int)((pos_x + constants.radius) / constants.diameter);
		if (tmp_cell > cell_x) {
			cells[4 * g_id + l_id] = (((tmp_cell << constants.x_shift) | cell_y) << 1) | 0x01;
			objects[4 * g_id + l_id] = g_id << 1 | 0x00;
			l_id++;
			count++;
			addValX = 1;
		}
		tmp_cell = (unsigned int)((pos_x - constants.radius) / constants.diameter);
		if (tmp_cell < cell_x) {
			cells[4 * g_id + l_id] = (((tmp_cell << constants.x_shift) | cell_y) << 1) | 0x01;
			objects[4 * g_id + l_id] = g_id << 1 | 0x00;
			l_id++;
			count++;
		}
		tmp_cell = (unsigned int)((pos_y + constants.radius) / constants.diameter);
		if (tmp_cell > cell_y) {
			cells[4 * g_id + l_id] = (((cell_x << constants.x_shift) | tmp_cell) << 1) | 0x01;
			objects[4 * g_id + l_id] = g_id << 1 | 0x00;
			l_id++;
			count++;
			addValY = 1;
		}
		tmp_cell = (unsigned int)((pos_y - constants.radius) / constants.diameter);
		if (tmp_cell < cell_y) {
			cells[4 * g_id + l_id] = (((cell_x << constants.x_shift) | tmp_cell) << 1) | 0x01;
			objects[4 * g_id + l_id] = g_id << 1 | 0x00;
			l_id++;
			count++;
		}
		dist = (pos_x - (cell_x * constants.diameter + constants.radius)) * (pos_x - (cell_x * constants.diameter + constants.radius)) + (pos_y - (cell_y * constants.diameter + constants.radius)) * (pos_y - (cell_y * constants.diameter + constants.radius));
		if (dist > constants.max_distance_squared) {
			cells[4 * g_id + l_id] = ((((cell_x + addValX) << constants.x_shift) | (cell_y + addValY)) << 1) | 0x01;
			objects[4 * g_id + l_id] = g_id << 1 | 0x00;
			count++;
		}
		else {
			cells[4 * g_id + l_id] = UINT_MAX;
			objects[4 * g_id + l_id] = g_id << 2;
		}
		if (l_id == 2) {
			cells[4 * g_id + 3] = UINT_MAX;
			objects[4 * g_id + 3] = g_id << 2;
		}

		g_id += blockDim.x * gridDim.x;
	}
	s[threadIdx.x] = count;
	__syncthreads();
	dSum(s, cell_count);
}

__global__ void radixSetup(unsigned int* radices, unsigned int* cells, const unsigned int shift, ConstantsRadixSetUp constants) {
	extern __shared__ unsigned int s[];
	int l_group = threadIdx.x / constants.threads_per_group;
	int group_start = (blockIdx.x * constants.groups_per_block + l_group) * constants.cells_per_group;
	int group_end = group_start + constants.cells_per_group;
	int index;
	for (int i = threadIdx.x; i < constants.shared_memory_size; i += blockDim.x) {
		s[i] = 0;
	}
	__syncthreads();
	for (int i = group_start + (threadIdx.x % constants.threads_per_group); (i < group_end) && (i < constants.cells_size); i += constants.threads_per_group) {
		index = ((cells[i] >> shift) & (constants.num_radices - 1)) + l_group * constants.num_radices;
		for (int j = 0; j < constants.threads_per_group; j++) {
			if (threadIdx.x % constants.threads_per_group == j) {
				s[index]++;
			}
		}
	}
	__syncthreads();
	for (int i = threadIdx.x; i < constants.shared_memory_size; i += blockDim.x) {
		radices[((i % constants.num_radices) * gridDim.x * constants.groups_per_block) + (blockIdx.x * constants.groups_per_block + i / constants.num_radices)] = s[i];//((i % constants.num_radices) * gridDim.x * constants.groups_per_block) + (blockIdx.x * constants.groups_per_block + i / constants.num_radices);
	}
}
__global__ void radixSum(unsigned int* radices, unsigned int* radices_prefix_sum, ConstantsRadixSum constants) {
	extern __shared__ unsigned int s[];
	int left = 0;
	int total;
	int empty_index;
	for (int j = 0; j < constants.radices_per_block && (blockIdx.x * constants.radices_per_block + j) < constants.num_radices; j++) {
		for (int i = threadIdx.x; i < constants.num_groups; i += blockDim.x) {
			s[i] = radices[blockIdx.x * constants.num_groups * constants.radices_per_block + j * constants.num_groups + i];
		}
		__syncthreads();
		empty_index = threadIdx.x + constants.num_groups;
		if (empty_index < constants.padded_groups) {
			s[empty_index] = 0;
		}
		__syncthreads();

		if (!threadIdx.x) {
			total = s[constants.num_groups - 1];
		}
		d_prefix_sum(s, constants.padded_groups);

		__syncthreads();

		for (int i = threadIdx.x; i < constants.num_groups; i += blockDim.x) {
			radices[blockIdx.x * constants.num_groups * constants.radices_per_block + j * constants.num_groups + i] = s[i];
		}

		__syncthreads();

		if (!threadIdx.x) {
			total += s[constants.num_groups - 1];
			radices_prefix_sum[blockIdx.x * constants.radices_per_block + j] = left;
			left += total;
		}
	}
	__syncthreads();
	if (!threadIdx.x) {
		radices_prefix_sum[constants.num_radices + blockIdx.x] = left;
	}
}
__global__ void radixReorder(unsigned int* radices_prefix_sum, unsigned int* radices, unsigned int* cells, unsigned int* objects, unsigned int* cells_out, unsigned int* objects_out, const unsigned int shift, ConstantsRadixReorder constants) {
	extern __shared__ unsigned int s[];
	unsigned int* t = s + constants.num_radices;
	int l_group = threadIdx.x / constants.threads_per_group;
	int group_start = (blockIdx.x * constants.groups_per_block + l_group) * constants.cells_per_group;
	int group_end = group_start + constants.cells_per_group;
	int index;
	for (int i = threadIdx.x; i < constants.num_radices; i += blockDim.x) {
		s[i] = radices_prefix_sum[i];
		if (i < gridDim.x) {
			t[i] = radices_prefix_sum[constants.num_radices + i];
		}
	}
	__syncthreads();
	for (int i = threadIdx.x + gridDim.x; i < constants.padded_blocks; i += blockDim.x) {
		t[i] = 0;
	}
	d_prefix_sum(t, constants.padded_blocks);
	__syncthreads();
	for (int i = threadIdx.x; i < constants.num_radices; i += blockDim.x) {
		s[i] += t[i / constants.radices_per_block];
	}
	__syncthreads();
	////can be deleted
	//if (!blockIdx.x) {
	//	for (int i = threadIdx.x; i < constants.num_radices; i += blockDim.x) {
	//		radices_prefix_sum_help[i] = s[i];
	//	}
	//}
	for (int i = threadIdx.x; i < constants.groups_per_block * constants.num_radices; i += blockDim.x) {
		//radices[blockIdx.x * constants.groups_per_block * constants.num_radices + i] = 5; s[i / constants.groups_per_block];//((i % constants.num_radices) * gridDim.x * constants.groups_per_block) + (blockIdx.x * constants.groups_per_block + i / constants.num_radices);
		//radices_help[blockIdx.x * constants.groups_per_block * constants.num_radices + i] = s[i % constants.num_radices] + radices[((i % constants.num_radices) * gridDim.x * constants.groups_per_block) + (blockIdx.x * constants.groups_per_block + i / constants.num_radices)];

		//this have to be uncomented!!!
		t[i] = s[i % constants.num_radices] + radices[((i % constants.num_radices) * gridDim.x * constants.groups_per_block) + (blockIdx.x * constants.groups_per_block + i / constants.num_radices)];
	}
	//uncomment!!!
	__syncthreads();
	for (int i = group_start + (threadIdx.x % constants.threads_per_group); (i < group_end) && (i < constants.cells_size); i += constants.threads_per_group) {
		index = ((cells[i] >> shift) & (constants.num_radices - 1)) + l_group * constants.num_radices;
		for (int j = 0; j < constants.threads_per_group; j++) {
			if (threadIdx.x % constants.threads_per_group == j) {
				cells_out[t[index]] = cells[i];
				objects_out[t[index]] = objects[i];
				t[index]++;
			}
		}
	}
}

__global__ void cellColide(unsigned int* cells, unsigned int* objects, unsigned int cells_per_thread, Particle* particles, unsigned int* collisions_count, unsigned int cell_count, ConstantsCellColide constants, unsigned int* particles_coliding_indices) {
	extern __shared__ unsigned int s[];
	if (!threadIdx.x) { s[0] = 0; }
	unsigned int thread_start = (blockIdx.x * blockDim.x + threadIdx.x) * cells_per_thread;
	unsigned int thread_end = thread_start + cells_per_thread;
	unsigned int last = UINT_MAX;
	unsigned int i = thread_start;
	unsigned int home_cell_count;
	unsigned int phantom_cell_count;
	unsigned int current_home;
	unsigned int current_phantom;
	int cell_start_index = -1;
	float dist;
	float dx;
	float dy;
	unsigned int collisions = 0;
	unsigned int id = 0;
	float dist2;
	float2 pos_a, pos_b;
	float temp;
	while (1) {
		if (i >= cell_count || cells[i] >> 1 != last) {
			if (cell_start_index + 1 && home_cell_count > 0 && (home_cell_count + phantom_cell_count) > 1) {
				for (int j = cell_start_index; j < cell_start_index + home_cell_count; j++) {
					current_home = objects[j] >> 1;
					
					for (int k = j + 1; k < i; k++) {
						current_phantom = objects[k] >> 1;
						pos_a.x = particles[current_phantom].position.x;
						pos_a.y = particles[current_phantom].position.y;
						pos_b.x = particles[current_home].position.x;
						pos_b.y = particles[current_home].position.y;
						//dx = particles[current_phantom].position.x - particles[current_home].position.x;
						//dy = particles[current_phantom].position.y - particles[current_home].position.y;
						dx = pos_a.x - pos_b.x + 0.000001;
						dy = pos_a.y - pos_b.y + 0.000001;
						dist = dx * dx + dy * dy;
						if (dist < constants.diameter_squared) {
							//id = atomicAdd(&s[0], 1);
							/*if ((current_phantom < current_home) && (k > cell_start_index + home_cell_count)) {
								continue;
							}*/
							//atomicAdd(&collisions, 1);
							//if (id < 1000) {
							//	atomicExch(&particles_coliding_indices[blockIdx.x * 2000 + 2 * id], current_home);
							//	atomicExch(&particles_coliding_indices[blockIdx.x * 2000 + 2 * id + 1], current_phantom);
							//	/*particles_coliding_indices[blockIdx.x * 2000 + 2 * id] = current_home;
							//	particles_coliding_indices[blockIdx.x * 2000 + 2 * id + 1] = current_phantom;*/
							//} 
							/*particles[current_home].position.x = 0.0f;
							particles[current_home].position.y = 0.0f;
							particles[current_phantom].position.x = 0.0f;
							particles[current_phantom].position.y = 0.0f;*/
							dist = sqrt(dist);
							dx /= dist;
							dy /= dist;
							pos_b.x -= dx * (3 * constants.radius - dist) / 2;
							pos_b.y -= dy * (3 * constants.radius - dist) / 2;

							pos_a.x += dx * (3 * constants.radius - dist) / 2;
							pos_a.y += dy * (3 * constants.radius - dist) / 2;
							pos_a.x = fmaxf(fminf(pos_a.x, constants.scene_width), constants.radius);
							pos_a.y = fmaxf(fminf(pos_a.y, constants.scene_hieght), constants.radius);
							pos_b.x = fmaxf(fminf(pos_b.x, constants.scene_width), constants.radius);
							pos_b.y = fmaxf(fminf(pos_b.y, constants.scene_hieght), constants.radius);
							atomicExch(&particles[current_phantom].position.x, pos_a.x);
							atomicExch(&particles[current_home].position.x, pos_b.x);

							atomicExch(&particles[current_phantom].position.y, pos_a.y);
							atomicExch(&particles[current_home].position.y, pos_b.y);

							//temp = particles[current_phantom].velocity.x;
							//atomicExch(&particles[current_phantom].velocity.x, particles[current_home].velocity.x);
							//atomicExch(&particles[current_home].velocity.x, temp);

							//// Swap velocity.y
							//temp = particles[current_phantom].velocity.y;
							//atomicExch(&particles[current_phantom].velocity.y, particles[current_home].velocity.y);
							//atomicExch(&particles[current_home].velocity.y, temp);
							//dist -= dist;
							//atomicAdd(&particles[current_home].position.x, -dx * constants.radius);
							//atomicAdd(&particles[current_phantom].position.x, dx * constants.radius);

							//atomicAdd(&particles[current_home].position.y, -dy * constants.radius);
							//atomicAdd(&particles[current_phantom].position.y, dy * constants.radius);
							/*particles[current_home].position.x -= dx * constants.radius;
							particles[current_phantom].position.x += dx * constants.radius;*/

							/*particles[current_home].position.y -= dy * constants.radius;
							particles[current_phantom].position.y += dy * constants.radius;*/
							//collisions++;
						}
					}
				}
			}

			if (i > thread_end || i >= cell_count) {
				break;
			}
			if (i != thread_start || !blockIdx.x && !threadIdx.x) {
				home_cell_count = 0;
				phantom_cell_count = 0;
				cell_start_index = i;

			}
			last = cells[i] >> 1;
		}
		if (cell_start_index + 1) {
			if (objects[i] & 0x01) {
				home_cell_count++;
			}
			else {
				phantom_cell_count++;
			}
		}
		i++;
	}
	//__syncthreads();
	//s[threadIdx.x] = collisions;
	//dSum(s, collisions_count);
}

void CollisionDetection::check_collision() {
	initData << <num_blocks, num_threads, sizeof(unsigned int)* num_threads >> > (d_particles, d_cells, d_objects, d_cell_count, constantsInitData);

	//unsigned int * h_cells = new unsigned int[cells_size];
	//cudaMemcpy(h_cells, d_cells, sizeof(unsigned int) * cells_size, cudaMemcpyDeviceToHost);
	//float* pos_x = new float[objects_size];
	//float* pos_y = new float[objects_size];
	//cudaMemcpy(pos_x, this->d_positions_x, sizeof(float) * objects_size, cudaMemcpyDeviceToHost);
	//cudaMemcpy(pos_y, this->d_positions_y, sizeof(float) * objects_size, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < objects_size; i++) {
	//	printf("%i\t%i\t%i\t%i\t%i\t%f\t%f\n",i, h_cells[i * 4], h_cells[i * 4 + 1], h_cells[i * 4 + 2], h_cells[i * 4 + 3], pos_x[i], pos_y[i]);
	//	if (h_cells[i * 4] == 0 || h_cells[i * 4 + 1] == 0 || h_cells[i * 4 + 2] == 0 || h_cells[i * 4 + 3] == 0) {
	//		int zzz = 2;
	//	}
	//}

	for (int shift = 0; shift < this->min_bits_for_hash; shift += this->bit_step_size) {
		radixSetup << <num_blocks, num_threads, sizeof(unsigned int)* shared_memory_size >> > (this->d_radices, this->d_cells, shift, constantsRadixSetUp);
		radixSum << <num_blocks, num_threads, sizeof(unsigned int)* padded_groups >> > (this->d_radices, this->d_radices_prefix_sum, constantsRadixSum);
		radixReorder << <num_blocks, num_threads, sizeof(unsigned int)* (shared_memory_size + num_radices) >> > (this->d_radices_prefix_sum, this->d_radices, this->d_cells, this->d_objects, this->d_cells_tmp, this->d_objects_tmp, shift, constantsRadixReorder);

		this->cells_swap = this->d_cells;
		this->d_cells = this->d_cells_tmp;
		this->d_cells_tmp = this->cells_swap;

		this->objects_swap = this->d_objects;
		this->d_objects = this->d_objects_tmp;
		this->d_objects_tmp = this->objects_swap;
	}

	/*unsigned int* h_cells = new unsigned int[cells_size];
	unsigned int* h_objects = new unsigned int[cells_size];
	cudaMemcpy(h_cells, this->d_cells, sizeof(float) * cells_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_objects, this->d_objects, sizeof(float) * cells_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < cells_size; i++) {
		printf("%i\t%i\n", h_cells[i], h_objects[i]);
	}*/


	cudaMemcpy(&this->h_cell_count, d_cell_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	this->cells_per_thread = (h_cell_count - 1) / (num_blocks * num_threads) + 1;
	

	cellColide << <num_blocks, num_threads, sizeof(unsigned int)* shared_size >> > (d_cells, d_objects, cells_per_thread, d_particles, d_collisions_count, h_cell_count, constantsCellColide, d_particles_coliding_indices);
	//cudaMemcpy(&this->collisions_count, d_collisions_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//cudaMemset(d_collisions_count, 0, sizeof(unsigned int));
	cudaMemset(d_cell_count, 0, sizeof(unsigned int));
	//printf("%u\n", this->collisions_count);
	//if (collisions_count == 0) {
	//	Particle* h_particles = new Particle[objects_size];
	//	unsigned int* h_cells = new unsigned int[cells_size];
	//	unsigned int* h_objects = new unsigned int[cells_size];
	//	cudaMemcpy(h_particles, this->d_particles, sizeof(Particle) * objects_size, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(h_cells, this->d_cells, sizeof(unsigned int) * cells_size, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(h_objects, this->d_objects, sizeof(unsigned int) * cells_size, cudaMemcpyDeviceToHost);

	//	for (int i = 1; i < cells_size; i++) {
	//		if (h_cells[i] > 2031616) {
	//			h_particles[h_objects[i] >> 1].position.x;
	//			Particle p1 = h_particles[h_objects[i] >> 1];
	//			Particle p2 = h_particles[h_objects[i-1] >> 1];
	//			float d = sqrt(pow(p1.position.x - p2.position.x, 2) + pow(p1.position.y - p2.position.y, 2));
	//			if (d < radius && d > 0.0f) {
	//				//count++;
	//				printf("%f\n", d);
	//			}
	//			int a = 2;
	//		}
	//	}
	//	int count = 0;
	//	for (int i = 0; i < objects_size; i++) {
	//		for (int j = i + 1; j < objects_size; j++) {
	//			Particle p1 = h_particles[i];
	//			Particle p2 = h_particles[j];
	//			float d = sqrt(pow(p1.position.x - p2.position.x, 2) + pow(p1.position.y - p2.position.y, 2));
	//			if (d < radius) {
	//				count++;
	//				printf("%f\n", d);
	//			}
	//		}
	//	}
	//	int adas = 2;
	//}
	//unsigned int* particles_coliding_indices = new unsigned int[2000 * num_blocks];
	//Particle* h_particles = new Particle[objects_size];
	//cudaMemcpy(particles_coliding_indices, this->d_particles_coliding_indices, sizeof(unsigned int) * 2000 * num_blocks, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_particles, this->d_particles, sizeof(Particle) * objects_size, cudaMemcpyDeviceToHost);
	//printf("sorted cells:\n");
	//for (int i = 0; i < 1000 * num_blocks; i++) {
	//	Particle p1, p2;
	//	p1.position = h_particles[particles_coliding_indices[2 * i]].position;
	//	p2.position = h_particles[particles_coliding_indices[2 * i + 1]].position;
	//	float d = sqrt(pow(p1.position.x - p2.position.x, 2) + pow(p1.position.y - p2.position.y, 2));
	//	if ((particles_coliding_indices[2 * i] == 21915 || particles_coliding_indices[2 * i] == 44088) || (particles_coliding_indices[2 * i + 1] == 21915 || particles_coliding_indices[2 * i + 1] == 44088)) {
	//	//if (d > 0 && d < 0.399) {
	//		//printf("%-5i: %-10f %f --- %-10f %f distance: %f id1:%i, id2 : %i\n", i, p1.position.x, p1.position.y, p2.position.x, p2.position.y, d, particles_coliding_indices[2 * i], particles_coliding_indices[2 * i + 1]);
	//	}
	//	//}
	//}
	//if(collisions_count < 1) {
	//	int abc = 2;
	//}
	//it++;
	//printf("--------------------------------------------------------------------------------\n");
	//printf("--------------------------------------------------------------------------------\n");
	//printf("--------------------------------------------------------------------------------\n");
}

void CollisionDetection::setup(float scene_width, float scene_height, unsigned int size, float radius, Particle* d_particles)
{
	this->zero = 0;
	this->objects_size = size;
	this->scene_width = scene_width;
	this->scene_height = scene_height;
	this->radius = radius;
	this->diameter = 2.0f * this->radius;
	this->width = ceilf(this->scene_width / this->diameter);
	this->height = ceilf(this->scene_height / this->diameter);
	unsigned int min_bits_for_width = count_bits(width);
	unsigned int min_bits_for_height = count_bits(height);
	this->min_bits_for_hash = min_bits_for_height + min_bits_for_width + 1;

	int deviceId;
	cudaGetDevice(&deviceId);
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, deviceId);
	shared_size = device_prop.sharedMemPerBlock / 4;
	this->num_blocks = 2 * device_prop.multiProcessorCount;
	this->threads_per_group = device_prop.warpSize;
	int num_passes = 1;
	bool tooMuchThreads = false;
	int shared_needed;
	while (1) {
		this->bit_step_size = (min_bits_for_hash - 1) / num_passes + 1;
		shared_needed = pow(2, this->bit_step_size);

		if (shared_needed < shared_size) {
			this->groups_per_block = (shared_size) / shared_needed;
			while (this->threads_per_group * this->groups_per_block >= device_prop.maxThreadsPerBlock) {
				tooMuchThreads = true;
				this->groups_per_block--;
			}
			break;
		}
		else {
			num_passes++;
		}
	}
	if (!tooMuchThreads) {
		this->groups_per_block--;
	}

	this->cells_size = 4 * this->objects_size;
	this->diameter_squared = this->diameter * this->diameter;
	this->max_distance_squared = pow(this->diameter * 0.7071067f - this->radius, 2);
	this->num_radices = pow(2, this->bit_step_size);
	this->num_threads = this->groups_per_block * this->threads_per_group;
	this->num_groups = this->num_blocks * this->groups_per_block;
	this->padded_groups = succesive_power_of_two(this->num_groups);
	this->padded_blocks = succesive_power_of_two(this->num_blocks);
	this->part_size = (this->objects_size - 1) / num_groups + 1;
	this->radices_per_block = (this->num_radices - 1) / this->num_blocks + 1;
	this->radices_size = this->num_groups * this->num_radices;
	this->cells_per_group = (cells_size - 1) / num_groups + 1;
	this->shared_memory_size = this->groups_per_block * this->num_radices;
	this->x_shift = min_bits_for_height;

	printf("bit_step_size: %u, num_blocks: %u, groups_per_block: %u, threads_per_group: %u, x_shift: %u, min_bits_for_hash: %u, diameter_squaredr: %f\n",
		this->bit_step_size, this->num_blocks, this->groups_per_block, this->threads_per_group, this->x_shift, this->min_bits_for_hash, this->diameter_squared);

	this->d_particles = d_particles;

	constantsInitData.diameter = this->diameter;
	constantsInitData.max_distance_squared = this->max_distance_squared;
	constantsInitData.radius = this->radius;
	constantsInitData.x_shift = this->x_shift;
	constantsInitData.objects_size = this->objects_size;

	constantsRadixSetUp.groups_per_block = this->groups_per_block;
	constantsRadixSetUp.threads_per_group = this->threads_per_group;
	constantsRadixSetUp.cells_per_group = this->cells_per_group;
	constantsRadixSetUp.shared_memory_size = this->shared_memory_size;
	constantsRadixSetUp.num_radices = this->num_radices;
	constantsRadixSetUp.cells_size = this->cells_size;

	constantsRadixSum.num_groups = this->num_groups;
	constantsRadixSum.num_radices = this->num_radices;
	constantsRadixSum.padded_groups = this->padded_groups;
	constantsRadixSum.radices_per_block = this->radices_per_block;

	constantsRadixReorder.cells_per_group = this->cells_per_group;
	constantsRadixReorder.cells_size = this->cells_size;
	constantsRadixReorder.groups_per_block = this->groups_per_block;
	constantsRadixReorder.num_radices = this->num_radices;
	constantsRadixReorder.padded_blocks = this->padded_blocks;
	constantsRadixReorder.radices_per_block = this->radices_per_block;
	constantsRadixReorder.threads_per_group = this->threads_per_group;

	constantsCellColide.diameter_squared = this->diameter_squared;
	constantsCellColide.radius = this->radius;
	constantsCellColide.objects_size = this->objects_size;
	constantsCellColide.scene_hieght = this->scene_height - this->diameter;
	constantsCellColide.scene_width = this->scene_width - this->diameter;

	cudaMalloc((void**)&this->d_radices, sizeof(unsigned int) * this->radices_size);
	cudaMalloc((void**)&this->d_radices_prefix_sum, sizeof(unsigned int) * (this->num_radices + this->num_blocks));
	cudaMalloc((void**)&this->d_cells, sizeof(unsigned int) * this->cells_size);
	cudaMalloc((void**)&this->d_cells_tmp, sizeof(unsigned int) * this->cells_size);
	cudaMalloc((void**)&this->d_objects, sizeof(unsigned int) * this->cells_size);
	cudaMalloc((void**)&this->d_objects_tmp, sizeof(unsigned int) * this->cells_size);
	cudaMalloc((void**)&this->d_cell_count, sizeof(unsigned int));
	cudaMalloc((void**)&this->d_collisions_count, sizeof(unsigned int));
	cudaMalloc((void**)&this->d_particles_coliding_indices, sizeof(unsigned int) * 2000 * num_blocks);
}

CollisionDetection::CollisionDetection(float width, float height, unsigned int size, float radius, Particle* d_particles)
{
	this->setup(width, height, size, radius, d_particles);
}

unsigned int CollisionDetection::succesive_power_of_two(unsigned int n) {
	if (n == 0)
		return 0;
	n = n - 1;
	n = n | (n >> 1);
	n = n | (n >> 2);
	n = n | (n >> 4);
	n = n | (n >> 8);
	n = n | (n >> 16);
	return n + 1;
}

unsigned int CollisionDetection::count_bits(unsigned int n) {
	unsigned int count = 0;
	while (n) {
		count++;
		n >>= 1;
	}
	return count;
}

