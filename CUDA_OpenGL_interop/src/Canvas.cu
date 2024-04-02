#include "Canvas.cuh"

__device__ float4 flying_ball(CanvasConstants constants)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	float x_norm = x / (float)constants.texture_resolution.x;
	float y_norm = y / (float)constants.texture_resolution.y;

	float t = constants.delta_time;
	float radius = 0.1;
	float2 point = make_float2(sin(t) * 0.5 + 0.5, cos(2 * t) * 0.5 + 0.5);
	float grey_value = 0;
	float a = (point.x - x_norm) * (point.x - x_norm) + (point.y - y_norm) * (point.y - y_norm);

	if (sqrtf(a) < radius)
		grey_value = 1;
	return make_float4(grey_value, a, a, 1.0f);
}

__device__ UpdateFunctionPtr d_flying_ball_ptr = flying_ball;

__global__ void updateCanvas(UpdateFunctionPtr func, cudaSurfaceObject_t surfaceObj, CanvasConstants constants) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < constants.texture_resolution.x && y < constants.texture_resolution.y) {
		surf2Dwrite((*func)(constants), surfaceObj, x * sizeof(float4), y);
	}
}

void Canvas::setUpdateFunctions()
{
	update_functions = std::vector<UpdateFunctionPtr>(1, 0);
	UpdateFunctionPtr h_update_functoin;
	cudaMemcpyFromSymbol(&h_update_functoin, d_flying_ball_ptr, sizeof(UpdateFunctionPtr));
	update_functions[CanvasStyle::gradient_id] = h_update_functoin;
}

void Canvas::init(glm::vec4&& boundings, glm::vec2&& texture_size, std::string&& shaders_path)
{
	this->boundings = std::move(boundings);
	this->texture_size = std::move(texture_size);
	shader = Shader(shaders_path + "/canvas.vert", shaders_path + "/canvas.frag");
	float quadVertices[] = {
		boundings.x, boundings.w, 0.0f, 1.0f,
		boundings.x, boundings.y, 0.0f, 0.0f,
		boundings.z, boundings.w, 1.0f, 1.0f,
		boundings.z, boundings.y, 1.0f, 0.0f,
	};
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture_size.x, texture_size.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glClearTexImage(texture, 0, GL_RGBA, GL_FLOAT, &glm::vec4(0.5)[0]);
	glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	cudaGraphicsResource_t cudaTextureResource;
	cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	cudaArray* cudaArray;
	cudaGraphicsMapResources(1, &cudaTextureResource, 0);
	cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaTextureResource, 0, 0);

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cudaArray;
	cudaCreateSurfaceObject(&canvas, &resDesc);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	block_size = dim3(sqrt(deviceProp.maxThreadsPerBlock), sqrt(deviceProp.maxThreadsPerBlock));
	grid_size = dim3((texture_size.x + block_size.x - 1) / block_size.x, (texture_size.y + block_size.y - 1) / block_size.y);
	constants.delta_time = 0.0f;
	constants.texture_resolution = make_float2(texture_size.x, texture_size.y);
	setUpdateFunctions();
}

void Canvas::update()
{
	constants.delta_time = glfwGetTime();
	updateCanvas << <grid_size, block_size >> > (update_functions[CanvasStyle::gradient_id], canvas, constants);
}

void Canvas::draw()
{
	glBindTexture(GL_TEXTURE_2D, texture);
	shader.use();
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}
