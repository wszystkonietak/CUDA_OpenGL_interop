#pragma once
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <math.h>
#include <string>

enum {
	s_Basic = 0,
	s_SoftBody = 1,
	s_Particles = 2,
	triangles = GL_TRIANGLES,
	points = GL_POINTS,
	lines = GL_LINES,
	line_strip = GL_LINE_STRIP,
	line_loop = GL_LINE_LOOP,
};


struct CanvasVertex {
	float2 position;
	float2 tex_cord;
};

struct Particle {
	float2 position;
	float2 velocity;
};

struct Texture {
	unsigned int id;
	std::string type;
};

struct cudaSpring {
	float rest_length;
	ushort2 indices;
};

struct Vertex {
	Vertex() { }
	Vertex(float2 position, float2 texCoords = make_float2(0, 0), float2 velocity = make_float2(0, 0)) : position(position), texCoords(texCoords), velocity(velocity), mass(.05){}
	float2 position;
	float2 velocity;
	float2 texCoords;
	float mass;
};

struct cudaVertex {
	float2 position;
	float2 velocity;
	float2 force;
	__device__ cudaVertex& operator=(const Vertex& vertex) {
		position = vertex.position;
		velocity = vertex.velocity;
		force.x = 0.0f;
		force.y = 0.0f;
		return *this;
	}
};