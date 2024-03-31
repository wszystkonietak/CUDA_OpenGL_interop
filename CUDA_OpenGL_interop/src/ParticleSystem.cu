#include "ParticleSystem.cuh"

void ParticleSystem::init(float scene_width, float scene_height, float particles_radius, int num_particles)
{
	this->particle_radius = particles_radius;
	std::uniform_real_distribution<float> rand_height(4 * particles_radius, scene_height - 4 * particles_radius);
	std::uniform_real_distribution<float> rand_width(4 * particles_radius, scene_width - 4 * particles_radius);
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		//p.position = make_float2(((float)rand() / RAND_MAX) * (scene_width - 2 * min_cell_size) + min_cell_size, ((float)rand() / RAND_MAX) * (scene_height - 2 * min_cell_size) + min_cell_size);
		//p.position = make_float2(((float)rand() / RAND_MAX) * 2 - 1, ((float)rand() / RAND_MAX) * 2 - 1);
		p.position = make_float2(rand_width(gen), rand_height(gen));
		p.velocity = make_float2(0, 0);
		particles.push_back(p);
	}
	setupParticleSystem();
	size_t num_bytes;
	cudaGraphicsResource* cuda_vbo_resource;
	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, cuda_vbo_resource);
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource);
	collisions.setup(1, 1, particles.size(), particle_radius, d_particles);
}

void ParticleSystem::draw(Shader& shader)
{
	shader.use();
	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, particles.size());
	glBindVertexArray(0);
}

void ParticleSystem::update()
{
	collisions.check_collision();
	//updateParticlesKernel << <collisions.num_blocks, collisions.num_threads >> > (d_particles, particles.size(), Time::delta_time, particle_radius, 1.0f - particle_radius);
	cudaDeviceSynchronize();
}

void ParticleSystem::setupParticleSystem()
{
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
}
