//#include "ParticleSystem.hpp"
//
//void ParticleSystem::init(float scene_width, float scene_height, int num_particles)
//{
//	float min_cell_size = 0.2f;
//	for (int i = 0; i < num_particles; i++) {
//		Particle p;
//		//p.position = make_float2(((float)rand() / RAND_MAX) * (scene_width - 2 * min_cell_size) + min_cell_size, ((float)rand() / RAND_MAX) * (scene_height - 2 * min_cell_size) + min_cell_size);
//		p.position = make_float2(((float)rand() / RAND_MAX) * 2 - 1, ((float)rand() / RAND_MAX) * 2 - 1);
//		p.velocity = make_float2(0, 0);
//		particles.push_back(p);
//	}
//	setupParticleSystem();
//}
//
//void ParticleSystem::draw(Shader& shader)
//{
//	shader.use();
//	glBindVertexArray(VAO);
//	glDrawArrays(GL_POINTS, 0, particles.size());
//	glBindVertexArray(0); 
//}
//
//void ParticleSystem::update()
//{
//
//}
//
//void ParticleSystem::setupParticleSystem()
//{
//	glGenVertexArrays(1, &VAO);
//	glGenBuffers(1, &VBO);
//	glBindVertexArray(VAO);
//
//	glBindBuffer(GL_ARRAY_BUFFER, VBO);
//	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), &particles[0], GL_DYNAMIC_DRAW);
//
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, position));
//
//	glEnableVertexAttribArray(1);
//	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, velocity));
//
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//	glBindVertexArray(0);
//}
