#include "Water.cuh"

__global__ void create_solid_cells(cudaSurfaceObject_t SurfObj, int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 data;
	if (x < width && y < height) {
		data.x = (x < 1 || x > width - 2) ? 0 : 1;
		data.y = (y < 1 || y > height - 2) ? 0 : 1;
		// Write to output surface
		surf2Dwrite(data, SurfObj, x * sizeof(float2), y);
	}
}

__global__ void clear_grid(cudaSurfaceObject_t grid, uint2 resolution)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		surf2Dwrite(0.0f, grid, x * sizeof(float), y);
	}
}

__global__ void simulate_particles(Particle* particles, SimulateParticlesConstants constants) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < constants.particles_size) {
		
		particles[index].position.x += particles[index].velocity.x * constants.delta_time;
		particles[index].position.y += particles[index].velocity.y * constants.delta_time;

		if (particles[index].position.x < constants.boundings.x) {
			particles[index].position.x = constants.boundings.x;
			particles[index].velocity.x = -particles[index].velocity.x;
		}
		if (particles[index].position.x > constants.boundings.y) {
			particles[index].position.x = constants.boundings.y;
			particles[index].velocity.x = -particles[index].velocity.x;
		}
		if (particles[index].position.y < constants.boundings.z) {
			particles[index].position.y = constants.boundings.z;
			particles[index].velocity.y = -particles[index].velocity.y;
		}
		if (particles[index].position.y > constants.boundings.w) {
			particles[index].position.y = constants.boundings.w;
			particles[index].velocity.y = -particles[index].velocity.y;
		}
		index += blockDim.x + gridDim.x;
	}
}

__global__ void particles_to_grid(cudaSurfaceObject_t grid, Particle* particles, float2* grid_velocities, float2* sum_of_weights, ParticlesToGridConstants constants)
{
	extern __shared__ float2 s[];
	float2* s_grid_velocities = s;
	float* s_sum_of_weights = (float*)&s_grid_velocities[constants.num_cells];
	int x = threadIdx.x;
	int y = threadIdx.y;
	int index = x * blockDim.y + y;

	uint2 grid_idx, cell;
	float2 l_pos;
	float2 pos, delta, vel;
	while (index < constants.particles_size) {
		pos = particles[index].position;
		vel = particles[index].velocity;
		grid_idx.x = pos.x / constants.grid_size;
		grid_idx.y = pos.y / constants.grid_size;
		if (grid_idx.x = blockIdx.x || grid_idx.y = blockIdx.y)
		{
			l_pos.x = pos.x - grid_idx.x * constants.grid_size;
			l_pos.y = pos.y - grid_idx.y * constants.grid_size;
			cell.x = pos.x / constants.cell_size;
			cell.y = pos.y / constants.cell_size;


			delta.x = (pos.x - constants.cell_size * cell.x) / constants.cell_size;
			delta.y = (pos.y - constants.cell_size * cell.y) / constants.cell_size;

			weights.x = (1 - delta.x) * (1 - delta.y);
			weights.y = delta.x * (1 - delta.y);
			weights.z = (delta.x) * (delta.y);
			weights.w = (1 - delta.x) * (delta.y);

			//left bottom
			atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y].x, weights.x * vel.x);
			atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y].y, weights.x * vel.y);
			atomicAdd(&s_sum_of_weights[cell.x * resolution.y + cell.y], weights.x);
			//bottom right
			atomicAdd(&s_grid_velocities[(cell.x + 1) * resolution.y + cell.y].x, weights.y * vel.x);
			atomicAdd(&s_grid_velocities[(cell.x + 1) * resolution.y + cell.y].y, weights.y * vel.y);
			atomicAdd(&s_sum_of_weights[(cell.x + 1) * resolution.y + cell.y], weights.y);
			//right top
			atomicAdd(&s_grid_velocities[(cell.x + 1) * resolution.y + cell.y + 1].x, weights.z * vel.x);
			atomicAdd(&s_grid_velocities[(cell.x + 1) * resolution.y + cell.y + 1].y, weights.z * vel.y);
			atomicAdd(&s_sum_of_weights[(cell.x + 1) * resolution.y + cell.y + 1], weights.z);
			//left top
			atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y + 1].x, weights.w * vel.x);
			atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y + 1].y, weights.w * vel.y);
			atomicAdd(&s_sum_of_weights[cell.x * resolution.y + cell.y + 1], weights.w);
			surf2Dwrite(1.0f, grid, cell.x * sizeof(float), cell.y);
		}
		index += blockDim.x;
	}
	__syncthreads();

	index = x * blockDim.y + y;

	while (x < constants.resolution.x || y < constants.resolution.y)
	{
		atomicAdd(&sum_of_weights[index], s_sum_of_weights[index]);
		atomicAdd(&grid_velocities[index].x, s_grid_velocities[index].x);
		atomicAdd(&grid_velocities[index].y, s_grid_velocities[index].y);

		x += blockDim.x;
		y += blockDim.y;
		index = x * blockDim.y + y;
	}
}

__global__ void update_velocities(cudaSurfaceObject_t grid, float2* grid_velocities, float* sum_of_weights)
{
	ushort2 coord;
	coord.x = blockIdx.x * blockDim.x + threadIdx.x;
	coord.y = blockIdx.y * blockDim.y + threadIdx.y;
	float2 vel;
	float sum;
	float is_busy;
	unsigned int id;
	if (coord.x < resolution.x && coord.y < resolution.y) {
		id = ((coord.x / 2) * resolution.y + coord.y) + black_cell_offset * ((coord.x + coord.y) % 2);
		sum = sum_of_weights[id];
		if (sum > 0.0) {
			surf2Dread(&is_busy, grid, coord.x * sizeof(float), coord.y);
			if (is_busy)
				busy_cells[atomicAdd(&busy_cells_size[0], 1)] = coord;
			grid_velocities[id].x /= sum;
			grid_velocities[id].y /= sum;
		}
	}
}

__global__ void calculate_divergence(cudaSurfaceObject_t solid_cells, float2* grid_velocities, float* sum_of_weights)
{

}

__global__ void simulate_particles(cudaSurfaceObject_t grid, Particle* particles, 
	float2* grid_velocities, float* sum_of_weights, unsigned int particles_size, uint2 resolution, 
	unsigned int grid_size, unsigned int black_cell_offset, float4 boundings, float cell_size, 
	float delta_time) 
{
	extern __shared__ float2 s[];
	float2* s_grid_velocities = s;
	float* s_sum_of_weights = (float*)&s_grid_velocities[grid_size];
	
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	int t_id = threadIdx.x, tex_id;
	float2 vel, pos, delta;
	uint2 cell = make_uint2(0, 0), g_cell;
	float4 weights, prev_weights;

	for (int i = t_id; i < grid_size; i += blockDim.x) {
		s_grid_velocities[i].x = 0;
		s_grid_velocities[i].y = 0;
		s_sum_of_weights[i] = 0;
	}

	__syncthreads();

	while (g_id < particles_size) {
		vel = particles[g_id].velocity;
		pos = particles[g_id].position;
		vel.y -= 9.81f * delta_time;
		
		pos.x += vel.x * delta_time;
		pos.y += vel.y * delta_time;

		if (pos.x <= boundings.x) {
			vel.x *= -0.8f;
			pos.x = boundings.x;
		}
		if (pos.x >= boundings.y) {
			vel.x *= -0.8f;
			pos.x = boundings.y;
		}

		if (pos.y <= boundings.z) {
			vel.y *= -0.8f;
			pos.y = boundings.z;
		}
		if (pos.y >= boundings.w) {
			vel.y *= -0.8f;
			pos.y = boundings.w;
		}

		particles[g_id].position = pos;
		cell = make_uint2(pos.x / cell_size, pos.y / cell_size);
		surf2Dwrite(1.0f, grid, cell.x * sizeof(float), cell.y);

		delta.x = (pos.x - cell_size * cell.x) / cell_size;
		delta.y = (pos.y - cell_size * cell.y) / cell_size;

		weights.x = (1 - delta.x) * (1 - delta.y);
		weights.y = delta.x * (1 - delta.y);
		weights.z = (delta.x) * (delta.y);
		weights.w = (1 - delta.x) * (delta.y);

		//left write
		atomicAdd(&s_grid_velocities[(cell.x -  1) * resolution.y + cell.y].x, weights.x * vel.x);
		//bottom write
		atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y - 1].y, weights.x * vel.y);
		//right write
		atomicAdd(&s_grid_velocities[(cell.x + 1) * resolution.y + cell.y].x, weights.y * vel.x);
		//top write
		atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y + 1].y, weights.y * vel.y);

		/*atomicAdd(&s_grid_velocities[(cell.x + 1) * resolution.y + cell.y + 1].x, weights.z * vel.x);
		atomicAdd(&s_grid_velocities[(cell.x + 1) * resolution.y + cell.y + 1].y, weights.z * vel.y);

		atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y + 1].x, weights.w * vel.x);
		atomicAdd(&s_grid_velocities[cell.x * resolution.y + cell.y + 1].y, weights.w * vel.y);*/

		atomicAdd(&s_sum_of_weights[cell.x * resolution.y + cell.y], weights.x);
		atomicAdd(&s_sum_of_weights[(cell.x + 1) * resolution.y + cell.y], weights.y);
		atomicAdd(&s_sum_of_weights[(cell.x + 1) * resolution.y + cell.y + 1], weights.z);
		atomicAdd(&s_sum_of_weights[cell.x * resolution.y + cell.y + 1], weights.w);

		g_id += blockDim.x * gridDim.x;
	}
	__syncthreads();

	g_cell.x = t_id / resolution.x;
	g_cell.y = t_id % resolution.y;
	tex_id = ((g_cell.x / 2) * resolution.y + g_cell.y) + (grid_size / 2) * ((g_cell.x + g_cell.y) % 2);

	while(g_cell.x < resolution.x)
	{
		atomicAdd(&sum_of_weights[tex_id], s_sum_of_weights[t_id]);
		atomicAdd(&grid_velocities[tex_id].x, s_grid_velocities[t_id].x);
		atomicAdd(&grid_velocities[tex_id].y, s_grid_velocities[t_id].y);
		
		t_id += blockDim.x;
		g_cell.x = t_id / resolution.x;
		g_cell.y = t_id % resolution.y;
		tex_id = ((g_cell.x / 2) * resolution.y + g_cell.y) + black_cell_offset * ((g_cell.x + g_cell.y) % 2);
	}
}

__global__ void update_velocities(cudaSurfaceObject_t grid, float2* grid_velocities, float* sum_of_weights, ushort2* busy_cells, unsigned int* busy_cells_size, uint2 resolution, unsigned int black_cell_offset)
{
	ushort2 coord;
	coord.x = blockIdx.x * blockDim.x + threadIdx.x;
	coord.y = blockIdx.y * blockDim.y + threadIdx.y;
	float sum;
	float2 vel;
	float is_busy;
	unsigned int id;
	if (coord.x < resolution.x && coord.y < resolution.y) {
		id = ((coord.x / 2) * resolution.y + coord.y) + black_cell_offset * ((coord.x + coord.y) % 2);
		sum = sum_of_weights[id];
		if (sum > 0.0) {
			surf2Dread(&is_busy, grid, coord.x * sizeof(float), coord.y);
			if (is_busy)
				busy_cells[atomicAdd(&busy_cells_size[0], 1)] = coord;
			vel = grid_velocities[id];
			vel.x /= sum;
			vel.y /= sum;
			grid_velocities[id] = vel;
		}
	}
}

__global__ void calculate_divergence(cudaSurfaceObject_t grid, cudaSurfaceObject_t solid_cells, float2* grid_velocities, uint2 resolution,
	ushort2* busy_cells, unsigned int* busy_cells_size, int num_iter,
	unsigned int grid_size, unsigned int black_cell_offset)
{
	//extern __shared__ float2 s_grid_velocities[];
	unsigned int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int it = threadIdx.x;
	ushort2 cell = busy_cells[g_id];
	
	unsigned int is_black = (cell.x + cell.y) % 2;
	unsigned int tex_write_id = ((cell.x / 2) * resolution.y + cell.y) + black_cell_offset * is_black;
	unsigned int tex_read_ids[4];
	tex_read_ids[0] = (((cell.x + 1) / 2) * resolution.y + cell.y) + black_cell_offset * is_black;
	tex_read_ids[1] = (((cell.x - 1) / 2) * resolution.y + cell.y) + black_cell_offset * is_black;
	tex_read_ids[2] = (((cell.x) / 2) * resolution.y + cell.y + 1) + black_cell_offset * is_black;
	tex_read_ids[3] = (((cell.x) / 2) * resolution.y + cell.y - 1) + black_cell_offset * is_black;
	float div = 0.0f;

	for (int i = threadIdx.x; i < 1; i++) 
	{
		float v1 = grid_velocities[tex_read_ids[0]].x;
		float v2 = grid_velocities[tex_read_ids[1]].x;
		float v3 = grid_velocities[tex_read_ids[2]].y;
		float v4 = grid_velocities[tex_read_ids[3]].y;
		div = v1 - v2 + v3 - v4;
	}
	printf("%f\n", div);
	if (g_id == 0)
	{
		printf("\n");
	}
	/*while (it < grid_size)
	{
		s_grid_velocities[it].x = grid_velocities[it].x;
		s_grid_velocities[it].y = grid_velocities[it].y;
		
		it += blockDim.x;
	}*/
}

__global__ void grid_to_particles(float2* grid_velocities, Particle* particles, float cell_size, int num_particles, uint2 resolution, unsigned int black_cell_offset)
{
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	float2 vel, pos, delta, vel_current, vel_right_bottom, vel_right_top, vel_left_top;
	uint2 cell;
	float4 weights;
	unsigned int tex_id;
	unsigned int is_black;
	while (g_id < num_particles) {
		pos = particles[g_id].position;
		cell.x = pos.x / cell_size;
		cell.y = pos.y / cell_size;

		is_black = (cell.x + cell.y) % 2;

		tex_id = ((cell.x / 2) * resolution.y + cell.y) + black_cell_offset * is_black;
		vel_current = grid_velocities[tex_id];

		tex_id = (((cell.x + 1) / 2) * resolution.y + cell.y) + black_cell_offset * !is_black;
		vel_right_bottom = grid_velocities[tex_id];

		tex_id = (((cell.x + 1) / 2) * resolution.y + cell.y + 1) + black_cell_offset * is_black;
		vel_right_top = grid_velocities[tex_id];

		tex_id = ((cell.x / 2) * resolution.y + cell.y + 1) + black_cell_offset * !is_black;
		vel_left_top = grid_velocities[tex_id];

		delta.x = (pos.x - cell_size * cell.x) / cell_size;
		delta.y = (pos.y - cell_size * cell.y) / cell_size;

		weights.x = (1 - delta.x) * (1 - delta.y);
		weights.y = delta.x * (1 - delta.y);
		weights.z = (delta.x) * (delta.y);
		weights.w = (1 - delta.x) * (delta.y);

		vel.x = weights.x * vel_current.x;
		vel.x += weights.y * vel_right_bottom.x;
		vel.x += weights.z * vel_right_top.x;
		vel.x += weights.w * vel_left_top.x;

		vel.y = weights.x * vel_current.y;
		vel.y += weights.y * vel_right_bottom.y;
		vel.y += weights.z * vel_right_top.y;
		vel.y += weights.w * vel_left_top.y;

		particles[g_id].velocity = vel;

		g_id += blockDim.x * gridDim.x;
	}
}

void FlipFluid::init(std::string&& shaders_path)
{
	cell_size = 0.05;
	particle_radius = 0.0123;
	s_textures = Shader(shaders_path + "/canvas.vert", shaders_path + "/canvas.frag");
	s_particles = Shader(shaders_path + "/particles.vert", shaders_path + "/particles.frag");
	s_particles.use();
	s_particles.setMat4("u_projectionViewMatrix", glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f));
	s_particles.setFloat("u_inPixelDiameter", particle_radius * 800);
	if (boundings.x > boundings.y) {
		std::swap(boundings.x, boundings.y);
	}
	if (boundings.z > boundings.w) {
		std::swap(boundings.z, boundings.w);
	}
	size.x = boundings.y - boundings.x;
	size.y = boundings.w - boundings.z;
	resolution = make_uint2(size.x / cell_size, size.y / cell_size);
	particles_size = size.x * size.y * rest_particle_density;
	particle_boundings = make_float4(boundings.x + cell_size + 0.00001, boundings.y - cell_size - 0.00001, boundings.z + cell_size + 0.00001, boundings.w - cell_size - 0.00001);
	mem_size = resolution.x * resolution.y;

	glGenTextures(1, &id_solid_cells);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id_solid_cells);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, resolution.x, resolution.y, 0, GL_RG, GL_FLOAT, NULL);
	glBindImageTexture(0, id_solid_cells, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);

	glGenTextures(1, &id_grid);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, id_grid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, resolution.x, resolution.y, 0, GL_RED, GL_FLOAT, NULL);
	//glClearTexImage(id_grid, 0, GL_RGBA, GL_FLOAT, &glm::vec1(0.5)[0]);
	glBindImageTexture(0, id_grid, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
	
	cudaMalloc(&d_busy_cells, sizeof(ushort2) * mem_size);
	cudaMalloc(&d_busy_cells_size, sizeof(unsigned int));
	h_busy_cells = new ushort2[mem_size];
	h_busy_cells_size = new unsigned int[1];

	cudaMalloc(&d_grid_velocities, sizeof(float2) * mem_size);
	cudaMalloc(&d_sum_of_weights, sizeof(float) * mem_size);


	solid_cells = Surface<float2>(id_solid_cells, GL_TEXTURE_2D, resolution);
	grid = Surface<float>(id_grid, GL_TEXTURE_2D, resolution);

	
	block_size = dim3(32, 32);
	grid_size = dim3(((unsigned int)resolution.x + block_size.x - 1) / block_size.x, 
		((unsigned int)resolution.y + block_size.y - 1) / block_size.y);
	p_block_size = dim3(1024);
	p_grid_size = dim3((particles_size - 1) / p_block_size.x + 1);
	create_solid_cells<< <grid_size, block_size >> > 
		(solid_cells.surface, (unsigned int)resolution.x, (unsigned int)resolution.y);
	particles = std::vector<Particle>(particles_size);


	std::uniform_real_distribution<float> rand_width(particle_boundings.x, particle_boundings.y / 2);
	std::uniform_real_distribution<float> rand_height(particle_boundings.z, particle_boundings.w);
	std::uniform_real_distribution<float> rand_vel(-1, 1);
	for (int i = 0; i < particles_size; i++) {
		particles[i].position = make_float2(rand_width(gen), rand_height(gen));
		particles[i].velocity = make_float2(rand_vel(gen), rand_vel(gen));
	}

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), &particles[0], GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, position));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, velocity));

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	size_t num_bytes;
	cudaGraphicsResource* cuda_vbo_resource;
	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, cuda_vbo_resource);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource);
	//collisions.setup(size.x, size.y, particles.size(), particle_radius, d_particles);
	cudaDeviceSynchronize();
	simulate_particles_shared_size = sizeof(float2) * mem_size + 
		sizeof(float) * mem_size + sizeof(unsigned short) * mem_size;
	black_cell_offset = mem_size / 2;
}
void FlipFluid::update()
{
	cudaMemset(d_busy_cells, 0, sizeof(ushort2) * mem_size);
	cudaMemset(d_busy_cells_size, 0, sizeof(unsigned int));

	cudaMemset(d_grid_velocities, 0, sizeof(float2) * mem_size);
	cudaMemset(d_sum_of_weights, 0, sizeof(float) * mem_size);
	clear_grid << <grid_size, block_size >> > (grid.surface, resolution);
	
	simulate_particles <<< p_grid_size, p_block_size, simulate_particles_shared_size >>> (grid.surface, d_particles, d_grid_velocities, d_sum_of_weights, particles_size, resolution, mem_size, black_cell_offset, particle_boundings, cell_size, Time::delta_time);

	update_velocities << <grid_size, block_size >> > (grid.surface, d_grid_velocities, d_sum_of_weights, d_busy_cells, d_busy_cells_size, resolution, black_cell_offset);

	cudaMemcpy(&h_busy_cells_size[0], &d_busy_cells_size[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
	calculate_divergence<<<((h_busy_cells_size[0] % 1024) - 1) / (h_busy_cells_size[0] % 1024) + 1, h_busy_cells_size[0] % 1024 >>>(grid.surface, solid_cells.surface, d_grid_velocities, resolution,
		d_busy_cells, d_busy_cells_size, num_iter, mem_size, black_cell_offset);

	grid_to_particles << <p_grid_size, p_block_size >> > (d_grid_velocities, d_particles, cell_size, particles_size, resolution, black_cell_offset);
}

void FlipFluid::draw()
{
	cudaDeviceSynchronize();
	s_textures.use();
	unsigned int quadvao = 0, quadvbo = 0;
	if (quadvao == 0) {
		float quadVertices[] = {
			// positions  // texture Coords
			-1.0f,  1.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadvao);
		glGenBuffers(1, &quadvbo);
		glBindVertexArray(quadvao);
		glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	}
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id_grid);
	glBindVertexArray(quadvao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	s_particles.use();    
	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, particles.size());
	glBindVertexArray(0);
}
