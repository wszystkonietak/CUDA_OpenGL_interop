#pragma once

#include "DataTypes.hpp"
#include "Shader.hpp"
#include <vector>
#include <math.h>
#include "cuda_gl_interop.h"
//#include "CudaDeviceProp.hpp"

enum CanvasStyle {
	gradient_id = 0,
};

struct CanvasConstants {
	float delta_time;
	float2 texture_resolution;
};

typedef float4 (*UpdateFunctionPtr)(CanvasConstants);

__global__ void updateCanvas(UpdateFunctionPtr func, cudaSurfaceObject_t canvas, CanvasConstants constants);
__device__ float4 gradient(CanvasConstants constants);


//__global__ void updateCanvas(cudaSurfaceObject_t canvas, CanvasConstants constants);


class Canvas {
public:
	Canvas() = default;
	Canvas(glm::vec4&& boundings, glm::vec2&& texture_size, std::string&& shaders_path) { init(std::move(boundings), std::move(texture_size), std::move(shaders_path)); };
	void setUpdateFunctions();
	void init(glm::vec4&& boundings, glm::vec2&& texture_size, std::string&& shaders_path);
	void update();
	void draw();
	cudaSurfaceObject_t canvas;
	std::vector<UpdateFunctionPtr> update_functions;
	glm::vec4 boundings;
	glm::vec2 texture_size;
	CanvasStyle style;
	GLuint texture;
	Shader shader;
	GLuint VAO, VBO;
	CanvasConstants constants;
	dim3 block_size;
	dim3 grid_size;
};


