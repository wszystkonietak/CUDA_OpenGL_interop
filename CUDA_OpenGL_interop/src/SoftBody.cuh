#pragma once
#include "Mesh.hpp"
#include "cuda_gl_interop.h"
#include "Time.hpp"
#include <glm/gtc/random.hpp>
#include "FrameHandler.hpp"


struct ConstantsSimulateSoftBody {
	int springs_size;
	int vertices_size;
	float stiffness;
	float damping;
	float delta_time;
	float friction_coefficient;
	float mass;
};


class SoftBody : public Mesh {
public:
	SoftBody() {}
	SoftBody(glm::vec2 size, float_t default_rest_length, glm::vec2 start)
		: size(size), default_rest_length(default_rest_length), particle_radius(default_rest_length / 5.0f),
		  sticked_particle_id(UINT_MAX) {
		init(start);
	}
	void init(glm::vec2 start = glm::vec2(0, 0));
	void simulate();
	glm::uvec2 dim;
	glm::vec2 size;
	float_t particle_radius;
	unsigned int sticked_particle_id;
	float_t default_rest_length;
	std::vector<cudaSpring> springs;
	ConstantsSimulateSoftBody constants;
	cudaSpring* d_springs;
	Vertex* d_vertices;
};