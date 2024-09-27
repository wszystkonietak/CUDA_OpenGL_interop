#include "CollisionDetection.cuh"
#include <vector>
#include <stdio.h>
#include "RandomSeed.hpp"
#include "Time.hpp"
#include "Shader.hpp"
#include <glm/gtc/matrix_transform.hpp>

class FlipFluid {
public:
	FlipFluid(glm::vec4&& boundings, std::string&& shaders_path) : boundings(std::move(boundings)) { init(std::move(shaders_path)); }
	void init(std::string&& shaders_path);
	void update();
	void draw();
	glm::vec4 boundings;
	float4 particle_boundings;
	float cell_size;
	float particle_radius;
	unsigned int particles_size;
	unsigned int rest_particle_density = 500;
	unsigned int num_iters = 2;
	uint2 resolution;
	glm::vec2 size;
	std::vector<Particle> particles;
	CollisionDetection collisions;
	unsigned int mem_size, simulate_particles_shared_size;

	Surface<float2> solid_cells;
	Surface<float> grid;
	Particle* d_particles;
	ushort2* d_busy_cells;
	float2* d_grid_velocities;
	float* d_sum_of_weights;
	unsigned int* d_busy_cells_size;
	unsigned int id_solid_cells, id_grid;
	unsigned int VAO, VBO;
	Shader s_textures, s_particles;
	dim3 block_size;
	dim3 grid_size;
	dim3 p_block_size;
	dim3 p_grid_size;
};
