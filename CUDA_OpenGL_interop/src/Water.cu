#include "Water.cuh"

struct FlipFluidConstants
{
	float2 resolution;
};

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

__global__ void clear_grid(cudaSurfaceObject_t SurfObj, int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		surf2Dwrite(0.0f, SurfObj, x * sizeof(float), y);
	}
}

__global__ void simulate_particles(cudaSurfaceObject_t grid, cudaSurfaceObject_t velocity, cudaSurfaceObject_t sum_of_weights, Particle* particles, uint2* busy_cells, int num_particles, float4 boundings, float cell_size, uint2 resolution, unsigned int* busy_cells_size, float delta_time) {
	extern __shared__ unsigned int s[];
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	int t_id = threadIdx.x;
	float2 vel, pos, delta;
	uint2 cell = make_uint2(0, 0), g_cell;
	float4 weights, vy;
	
	for (int i = t_id; i < resolution.x * resolution.y; i += blockDim.x) {
		s[i] = 0;
	}

	while (g_id < num_particles) {
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

		s[cell.x * resolution.y + cell.y]++;

		delta.x = (pos.x - cell_size * cell.x) / cell_size;
		delta.y = (pos.y - cell_size * cell.y) / cell_size;

		weights.x = (1 - delta.x) * (1 - delta.y);
		weights.y = delta.x * (1 - delta.y);
		weights.z = (delta.x) * (delta.y);
		weights.w = (1 - delta.x) * (delta.y);

		surf2Dwrite(make_float2(weights.x * vel.x, weights.x * vel.y), velocity, cell.x * sizeof(float2), cell.y);
		surf2Dwrite(make_float2(weights.y * vel.x, weights.y * vel.y), velocity, (cell.x + 1) * sizeof(float2), cell.y);
		surf2Dwrite(make_float2(weights.z * vel.x, weights.z * vel.y), velocity, (cell.x + 1) * sizeof(float2), cell.y + 1);
		surf2Dwrite(make_float2(weights.w * vel.x, weights.w * vel.y), velocity, cell.x * sizeof(float2), cell.y + 1);

		surf2Dwrite(weights.x, sum_of_weights, cell.x * sizeof(float), cell.y);
		surf2Dwrite(weights.y, sum_of_weights, (cell.x + 1) * sizeof(float), cell.y);
		surf2Dwrite(weights.z, sum_of_weights, (cell.x + 1) * sizeof(float), cell.y + 1);
		surf2Dwrite(weights.w, sum_of_weights, cell.x * sizeof(float), cell.y + 1);

		g_id += blockDim.x * gridDim.x;
	}
	__syncthreads();

	g_cell.x = t_id / resolution.x;
	g_cell.y = t_id % resolution.y;

	while(g_cell.x < resolution.x)
	{
		if (s[t_id])
		{
			busy_cells[atomicAdd(&busy_cells_size[0], 1)] = g_cell;
			surf2Dwrite(1.0f, grid, g_cell.x * sizeof(float), g_cell.y);
		}

		t_id += blockDim.x;
		g_cell.x = t_id / resolution.x;
		g_cell.y = t_id % resolution.y;
	}
}

__global__ void update_velocities(cudaSurfaceObject_t velocity, cudaSurfaceObject_t sum_of_weights, uint2* busy_cells, unsigned int* busy_cells_size)
{
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = g_id; i < busy_cells_size[0]; i += blockDim.x * gridDim.x)
	{

	}
}

__global__ void grid_to_particles(cudaSurfaceObject_t velocity, Particle* particles, float cell_size, int num_particles)
{
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;
	float2 vel, pos, delta, vel_current, vel_right_bottom, vel_right_top, vel_left_top;
	int2 cell;
	float4 weights;

	while (g_id < num_particles) {

		pos = particles[g_id].position;
		cell = make_int2(pos.x / cell_size, pos.y / cell_size);

		surf2Dread(&vel_current, velocity, cell.x * sizeof(float2), cell.y);
		surf2Dread(&vel_right_bottom, velocity, (cell.x + 1) * sizeof(float2), cell.y);
		surf2Dread(&vel_right_top, velocity, (cell.x + 1) * sizeof(float2), cell.y + 1);
		surf2Dread(&vel_left_top, velocity, cell.x * sizeof(float2), cell.y + 1);

		delta.x = (pos.x - cell_size * cell.x) / cell_size;
		delta.y = (pos.y - cell_size * cell.y) / cell_size;

		weights.x = (1 - delta.x) * (1 - delta.y);
		weights.y = delta.x * (1 - delta.y);
		weights.z = (delta.x) * (delta.y);
		weights.w = (1 - delta.x) * (delta.y);

		vel.x = vel_current.x;
		vel.x += vel_right_bottom.x;
		vel.x += vel_right_top.x;
		vel.x += vel_left_top.x;

		vel.y = vel_current.y;
		vel.y += vel_right_bottom.y;
		vel.y += vel_right_top.y;
		vel.y += vel_left_top.y;

		particles[g_id].velocity = vel;

		g_id += blockDim.x * gridDim.x;
	}
}

void FlipFluid::init(std::string&& shaders_path)
{
	cell_size = 0.02;
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
	resolution = size / cell_size;
	particles_size = size.x * size.y * rest_particle_density;
	particle_boundings = make_float4(boundings.x + cell_size + 0.00001, boundings.y - cell_size - 0.00001, boundings.z + cell_size + 0.00001, boundings.w - cell_size - 0.00001);

	float2* h_solid_cells = new float2[resolution.x * resolution.y];
	for (int y = 0; y < resolution.y; y++) {
		for (int x = 0; x < resolution.x; x++) {
			float2 val = make_float2(0, 0);
			val.x = x < 1 || x > resolution.x - 2 ? 0 : 1;
			val.y = y < 1 || y > resolution.y - 2 ? 0 : 1;
			h_solid_cells[x * (int)resolution.y + y] = val;
		}
	}

	glGenTextures(1, &id_solid_cells);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id_solid_cells);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, resolution.x, resolution.y, 0, GL_RG, GL_FLOAT, NULL);
	glBindImageTexture(0, id_solid_cells, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);

	glGenTextures(1, &id_velocity);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, id_velocity);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, resolution.x, resolution.y, 0, GL_RG, GL_FLOAT, NULL);
	glBindImageTexture(0, id_velocity, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);

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

	cudaMalloc(&d_busy_cells, sizeof(uint2) * resolution.x * resolution.y);
	cudaMalloc(&d_busy_cells_size, sizeof(unsigned int));
	solid_cells = Surface<float2>(id_solid_cells, GL_TEXTURE_2D);
	velocity = Surface<float2>(id_velocity, GL_TEXTURE_2D);
	grid = Surface<float>(id_grid, GL_TEXTURE_2D);
	//velocity = Surface<float2>(make_float2(resolution.x, resolution.y));
	//grid = Surface<float>(make_float2(resolution.x, resolution.y));
	density = Surface<float>(make_float2(resolution.x, resolution.y));
	sum_of_weights = Surface<float>(make_float2(resolution.x, resolution.y));
	
	block_size = dim3(32, 32);
	grid_size = dim3(((unsigned int)resolution.x + block_size.x - 1) / block_size.x, ((unsigned int)resolution.y + block_size.y - 1) / block_size.y);
	p_block_size = dim3(1024);
	p_grid_size = dim3((particles_size - 1) / p_block_size.x + 1);
	create_solid_cells<< <grid_size, block_size >> > (solid_cells.surface, (unsigned int)resolution.x, (unsigned int)resolution.y);
	particles = std::vector<Particle>(particles_size);
	std::uniform_real_distribution<float> rand_width(boundings.x + cell_size, boundings.y - cell_size);
	std::uniform_real_distribution<float> rand_height(boundings.z + cell_size, boundings.w - cell_size);
	for (int i = 0; i < particles_size; i++) {
		particles[i].position = make_float2(rand_width(gen), rand_height(gen));
		particles[i].velocity = make_float2(0, 0);
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
}

void FlipFluid::update()
{
	cudaMemset(d_busy_cells_size, 0, 1);
	clear_grid << <grid_size, block_size >> > (grid.surface, resolution.x, resolution.y);
	//collisions.check_collision();

	simulate_particles << <p_grid_size, p_block_size, sizeof(unsigned int) * resolution.x * resolution.y >> > (grid.surface, velocity.surface, sum_of_weights.surface, d_particles, d_busy_cells, particles.size(), particle_boundings, cell_size, make_uint2(resolution.x, resolution.y), d_busy_cells_size, Time::delta_time);
	update_velocities << <grid_size, block_size >> > ();
	grid_to_particles << <p_grid_size, p_block_size >> > (velocity.surface, d_particles, cell_size, particles.size());
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
