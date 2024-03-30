//#include "Water.cuh"
//
//__global__ void updateParticlesKernel(Particle* particles, int numParticles, float deltatime, float radius, float scene_size) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (idx < numParticles) {
//		float newPos_x = particles[idx].position.x + particles[idx].velocity.x * deltatime;
//		float newPos_y = particles[idx].position.y + particles[idx].velocity.y * deltatime;
//
//		if (newPos_x < radius)
//			newPos_x = radius;
//		if (newPos_x > scene_size)
//			newPos_x = scene_size;
//
//		if (newPos_y < radius)
//			newPos_y = radius;
//		if (newPos_y > scene_size)
//			newPos_y = scene_size;
//
//		particles[idx].position.x = newPos_x;
//		particles[idx].position.y = newPos_y;
//	}
//}
//
//
//
//void Water::init(float scene_width, float scene_height, int num_particles)
//{
//	particle_radius = 0.0005f;
//	this->scene_width = scene_width;
//	float min_cell_size = 0.2f;
//	for (int i = 0; i < num_particles; i++) {
//		Particle p;
//		//p.position = make_float2(((float)rand() / RAND_MAX) * (scene_width - 2 * min_cell_size) + min_cell_size, ((float)rand() / RAND_MAX) * (scene_height - 2 * min_cell_size) + min_cell_size);
//		p.position = make_float2(((float)rand() / RAND_MAX) * (1 - 8 * particle_radius) + 4 * particle_radius, ((float)rand() / RAND_MAX) * (1 - 8*particle_radius) + 4 * particle_radius);
//		p.velocity = make_float2(((float)rand() / RAND_MAX) * 0.2 - 0.1, ((float)rand() / RAND_MAX) * 0.1 - 0.05);
//		particles.push_back(p);
//	}
//	setupParticleSystem();
//	size_t num_bytes;
//	cudaGraphicsResource* cuda_vbo_resource;
//	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsNone);
//	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
//	cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, cuda_vbo_resource);
//	cudaGraphicsUnmapResources(1, &cuda_vbo_resource);
//	collisions.setup(1, 1, particles.size(), particle_radius, d_particles);
//	/*collisions.check_collision();
//	collisions.check_collision();
//	collisions.check_collision();
//	collisions.check_collision();
//	collisions.check_collision();
//	collisions.check_collision();
//	collisions.check_collision();
//	collisions.check_collision();
//	collisions.check_collision();*/
//}
//
//void Water::update(float deltatime)
//{
//	deltatime /= 100;
//	collisions.check_collision();
//	updateParticlesKernel << <collisions.num_blocks, collisions.num_threads >> > (d_particles, particles.size(), deltatime, particle_radius, 1.0f-particle_radius);
//	cudaDeviceSynchronize();
//}
