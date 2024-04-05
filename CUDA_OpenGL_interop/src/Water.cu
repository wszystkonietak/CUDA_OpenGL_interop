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

__global__ void simulate_particles(Particle* particles, int num_particles, float4 boundings, float delta_time) {
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (g_id < num_particles) {

		// Apply gravity
		particles[g_id].velocity.y -= 9.81f * delta_time; // Adjust gravity value as needed

		// Update position
		particles[g_id].position.x += particles[g_id].velocity.x * delta_time;
		particles[g_id].position.y += particles[g_id].velocity.y * delta_time;

		if (particles[g_id].position.x < boundings.x) {
			particles[g_id].velocity.x *= -0.8f;
			particles[g_id].position.x = boundings.x;
		}
		if (particles[g_id].position.x > boundings.y) {
			particles[g_id].velocity.x *= -0.8f;
			particles[g_id].position.x = boundings.y;
		}

		if (particles[g_id].position.y < boundings.z) {
			particles[g_id].velocity.y *= -0.8f;
			particles[g_id].position.y = boundings.z;
		}
		if (particles[g_id].position.y > boundings.w) {
			particles[g_id].velocity.y *= -0.8f;
			particles[g_id].position.y = boundings.w;
		}
		
	}
}

__global__ void particles_to_grid(cudaSurfaceObject_t grid, Particle* particles, float cell_size, unsigned int particles_size)
{
	unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;
	int2 cell = make_int2(particles[t_id].position.x / cell_size, particles[t_id].position.y / cell_size);
	surf2Dwrite(1.0f, grid, cell.x * sizeof(float), cell.y);
}


void FlipFluid::init(std::string&& shaders_path)
{
	cell_size = 0.02;
	particle_radius = 0.005;
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
	collisions.setup(size.x, size.y, particles.size(), particle_radius, d_particles);
	particles_to_grid << <1, particles_size >> > (grid.surface, d_particles, cell_size, particles_size);
	cudaDeviceSynchronize();
}

void FlipFluid::update()
{
	particles_to_grid << <1, particles_size >> > (grid.surface, d_particles, cell_size, particles_size);
	collisions.check_collision();
	simulate_particles << <p_grid_size, p_block_size >> > (d_particles, particles.size(), make_float4(boundings.x, boundings.y, boundings.z, boundings.w), Time::delta_time);
}

void FlipFluid::draw()
{
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
