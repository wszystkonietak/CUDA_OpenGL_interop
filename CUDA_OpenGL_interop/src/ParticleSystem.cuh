#include "DataTypes.hpp"
#include "Shader.hpp"
#include "RandomSeed.hpp"
#include "Time.hpp"
#include "CollisionDetection.cuh"
#include "cuda_gl_interop.h"

class ParticleSystem {
public:
	ParticleSystem() = default;
	ParticleSystem(float scene_width, float scene_height, float particle_radius, int num_particles)
	{
		init(scene_width, scene_height, particle_radius, num_particles);
	}
	virtual void init(float scene_width, float scene_height, float particles_radius, int num_particles);
	virtual void draw(Shader& shader);
	virtual void update();
	void setupParticleSystem();
	float particle_radius;
	CollisionDetection collisions;
	Particle* d_particles;
	std::vector<Particle> particles;
	unsigned int VAO;
	unsigned int VBO;
};