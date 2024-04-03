#include "CollisionDetection.cuh"
#include <vector>
#include "RandomSeed.hpp"
#include "Shader.hpp"

class FlipFluid {
public:
	FlipFluid(glm::vec4&& boundings, std::string&& shaders_path) : boundings(std::move(boundings)) { init(std::move(shaders_path)); }
	void init(std::string&& shaders_path);
	void update();
	void draw();
	glm::vec4 boundings;
	float cell_size;
	float particle_radius;
	unsigned int particles_size;
	unsigned int rest_particle_density = 20000;
	glm::vec2 resolution;
	glm::vec2 size;
	std::vector<Particle> particles;
	unsigned int id_solid_cells;
	CollisionDetection collisions;
	Surface<float2> velocity;
	Surface<float2> solid_cells;
	Surface<float> grid;
	Surface<float> density;
	Surface<float> sum_of_weights;
	Particle* d_particles;
	unsigned int VAO, VBO;
	Shader shader;
};
