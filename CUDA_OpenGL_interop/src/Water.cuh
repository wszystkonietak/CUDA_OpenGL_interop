#include "CollisionDetection.cuh"
#include <vector>
#include "RandomSeed.hpp"
#include "Time.hpp"
#include "Shader.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

typedef uint unsigned int;

struct SimulateParticlesConstants 
{
	float delta_time;
	float cell_size;
	uint grid_size;
	uint particles_size;
	uint black_cell_offset;
	uint2 resolution;
	float4 boundings;
};

struct ParticlesToGridConstants
{
	float delta_time;
	float cell_size;
	uint grid_size;
	uint particles_size;
	uint cells_per_grid;
	uint black_cell_offset;
	uint2 resolution;
	float4 boundings;
};

struct UpdateVelocitiesConstants 
{
	uint black_cell_offset;
	uint2 resolution;
};

struct CalculateDivergenceConstants
{
	uint num_iter;
	uint grid_size;
	uint black_cell_offset;
	uint2 resolutioin;
};

struct GridToParticlesConstants
{
	float cell_size;
	uint num_particles;
	uint black_cell_offset;
	uint2 resolution;
};

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
	unsigned int rest_particle_density = 1;
	unsigned int num_iter = 10;
	uint2 resolution;
	glm::vec2 size;
	std::vector<Particle> particles;
	CollisionDetection collisions;
	unsigned int mem_size, simulate_particles_shared_size, black_cell_offset;
	Surface<float2> solid_cells;
	Surface<float> grid;
	Particle* d_particles;
	ushort2* d_busy_cells;
	ushort2* h_busy_cells;
	unsigned int* h_busy_cells_size;
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
	SimulateParticlesConstants simulate_particles_constants;
	ParticlesToGridConstants particles_to_grid_constants;
	UpdateVelocitiesConstants update_velocities_constants;
	CalculateDivergenceConstants calculate_divergence;
	GridToParticlesConstants gir_to_particles_constants;
};
