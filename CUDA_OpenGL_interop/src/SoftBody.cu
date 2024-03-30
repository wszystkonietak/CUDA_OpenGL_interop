#include "SoftBody.cuh"
__global__ void kernel(int* a, int* b, unsigned int N) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] += b[i];
    }
}
__global__ void printSprings(cudaSpring* arr, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		printf("Value %d: %u -- %u -- %f\n", idx, arr[idx].indices.x, arr[idx].indices.y, arr[idx].rest_length);
	}
}

__global__ void printValues(Vertex* arr, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		printf("Value %d: %-8f -- %-8f\t", idx, arr[idx].position.x, arr[idx].position.y);
	}
	if (!threadIdx.x) {
		printf("\n");
	}
}

__global__ void simulateSoftBody(const cudaSpring* springs, Vertex* vertices, ConstantsSimulateSoftBody constants) {
	extern __shared__ cudaSpring s[];
	cudaSpring* s_indices = s;
	cudaVertex* s_vertices = (cudaVertex*)&s_indices[constants.springs_size];
	float2 deltaPosition, relativeVelocity, normalizedDelta, force;
	ushort2 id;
	float dot_product, distance;
	for (int i = threadIdx.x; i < constants.springs_size; i += blockDim.x) {
		s_indices[i] = springs[i];
	}
	for (int i = threadIdx.x; i < constants.vertices_size; i += blockDim.x) {
		s_vertices[i] = vertices[i];
	}
	__syncthreads();
	for (int i = threadIdx.x; i < constants.springs_size; i += blockDim.x) {
		id = s_indices[i].indices;
		deltaPosition.x = s_vertices[id.y].position.x - s_vertices[id.x].position.x;
		deltaPosition.y = s_vertices[id.y].position.y - s_vertices[id.x].position.y;
		relativeVelocity.x = s_vertices[id.y].velocity.x - s_vertices[id.x].velocity.x;
		relativeVelocity.y = s_vertices[id.y].velocity.y - s_vertices[id.x].velocity.y;
		distance = sqrtf(deltaPosition.x * deltaPosition.x + deltaPosition.y * deltaPosition.y) + 0.0001;
		normalizedDelta.x = deltaPosition.x / distance;
		normalizedDelta.y = deltaPosition.y / distance;
		dot_product = relativeVelocity.x * normalizedDelta.x + relativeVelocity.y * normalizedDelta.y;
		force.x = constants.stiffness * (distance - s_indices[i].rest_length) * normalizedDelta.x + normalizedDelta.x * dot_product * constants.damping;
		force.y = constants.stiffness * (distance - s_indices[i].rest_length) * normalizedDelta.y + normalizedDelta.y * dot_product * constants.damping;
		atomicAdd(&(s_vertices[id.x].force.x), force.x);
		atomicAdd(&(s_vertices[id.x].force.y), force.y);
		atomicAdd(&(s_vertices[id.y].force.x), -force.x);
		atomicAdd(&(s_vertices[id.y].force.y), -force.y);
	}
	//__syncthreads();

	for (int i = threadIdx.x; i < constants.vertices_size; i += blockDim.x) {
		int a = 2;
		vertices[i].velocity.x += (s_vertices[i].force.x / constants.mass) * constants.delta_time;
		vertices[i].velocity.y += (s_vertices[i].force.y / constants.mass) * constants.delta_time;
		vertices[i].position.x += vertices[i].velocity.x * constants.delta_time;
		vertices[i].position.y += vertices[i].velocity.y * constants.delta_time;
	}
	for (int i = threadIdx.x; i < constants.vertices_size; i += blockDim.x) {
		if (vertices[i].position.x < 0.0f) {
			vertices[i].velocity.x *= -1;
			vertices[i].position.x = 0.0001;
		}
		if (vertices[i].position.y < 0.0f) {
			vertices[i].velocity.y *= -1;
			vertices[i].position.y = 0.0001;
		}
		if (vertices[i].position.x > 1.0f) {
			vertices[i].velocity.x *= -1;
			vertices[i].position.x = 0.999;
		}
		if (vertices[i].position.y > 1.0f) {
			vertices[i].velocity.y *= -1;
			vertices[i].position.y = 0.999;
		}
	}
}

void SoftBody::init(glm::vec2 start)
{
	dim = size / default_rest_length;
	for (int y = 0; y < dim.y; y++) {
		for (int x = 0; x < dim.x; x++) {
			glm::vec2 position(start.x + x * default_rest_length, start.y + y * default_rest_length);
			glm::vec2 texCoords(default_rest_length * x / size.x, default_rest_length * y / size.y);
			glm::vec2 velocity = glm::vec2(0); glm::linearRand(glm::vec2(-.15, -.05), glm::vec2(.00, .00));
			vertices.push_back(Vertex(make_float2(position.x, position.y), make_float2(texCoords.x, texCoords.y), make_float2(velocity.x, velocity.y)));
		}
	}
	
	for (int y = 0; y < dim.y; y++) {
		for (int x = 0; x < dim.x; x++) {
			int32_t idLeftTop = y * dim.x + x;
			int32_t idRightTop = y * dim.x + (x + 1);
			int32_t idRightBottom = (y + 1) * dim.x + (x + 1);
			int32_t idLeftBottom = (y + 1) * dim.x + x;
			if (y == (dim.y - 1) && x == (dim.x - 1)) { continue; }
			if (y == (dim.y - 1)) {
				indices.push_back(idLeftTop);
				indices.push_back(idRightTop);
				continue;
			}
			if (x == (dim.x - 1)) {
				indices.push_back(idLeftTop);
				indices.push_back(idLeftBottom);
				continue;
			}
			indices.push_back(idLeftTop);
			indices.push_back(idRightTop);
			indices.push_back(idLeftTop);
			indices.push_back(idRightBottom);
			indices.push_back(idLeftTop);
			indices.push_back(idLeftBottom);
			indices.push_back(idRightTop);
			indices.push_back(idLeftBottom);
		}
	}
	for (int i = 0; i < indices.size(); i += 2) {
		cudaSpring s;
		s.indices = make_ushort2(indices[i], indices[i + 1]);
		s.rest_length = sqrt(pow(vertices[indices[i]].position.x - vertices[indices[i + 1]].position.x, 2)
			+ pow(vertices[indices[i]].position.y - vertices[indices[i + 1]].position.y, 2));
		springs.push_back(s);
	}
	
	setupMesh();
	size_t num_bytes;
	cudaGraphicsResource* cuda_vbo_resource;
	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, cuda_vbo_resource);
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource);
	cudaMalloc((void**)&d_springs, springs.size() * sizeof(cudaSpring));
	cudaMemcpy(d_springs, &springs[0], springs.size() * sizeof(cudaSpring), cudaMemcpyHostToDevice);
	constants.damping = 4.8;
	constants.friction_coefficient = 0.1;
	constants.stiffness = 1000;
	constants.delta_time = 0.0051f;
	constants.springs_size = springs.size();
	constants.vertices_size = vertices.size();
	constants.mass = 1; 0.5 / (float)vertices.size();
	//printf("%i\n", springs.size() + vertices.size());
}

void SoftBody::simulate()
{
	//collisions.check_collision();
	constants.delta_time = Time::delta_time;
	//simulateSoftBody << <1, springs.size(), vertices.size() * sizeof(cudaVertex) + springs.size() * sizeof(cudaSpring) >> > (d_springs, d_vertices, constants);
	//cudaDeviceSynchronize();
	//printa << <1, 1 >> > (5);
	float2 deltaPosition, relativeVelocity, a, b, force, normalizedDelta;
	std::vector<float2> forces(100, make_float2(0, 0));
	for (auto& spring : springs) {
		Vertex* A = &vertices[spring.indices.x];
		Vertex* B = &vertices[spring.indices.y];
		deltaPosition.x = B->position.x - A->position.x;
		deltaPosition.y = B->position.y - A->position.y;
		relativeVelocity.x = B->velocity.x - A->velocity.x;
		relativeVelocity.y = B->velocity.y - A->velocity.y;
		float_t distance = sqrt(deltaPosition.x * deltaPosition.x + deltaPosition.y * deltaPosition.y);
		normalizedDelta.x = deltaPosition.x / distance;
		normalizedDelta.y = deltaPosition.y / distance;

		a.x = constants.stiffness * (distance - spring.rest_length) * normalizedDelta.x;
		a.y = constants.stiffness * (distance - spring.rest_length) * normalizedDelta.y;
		b.x = normalizedDelta.x * (relativeVelocity.x * normalizedDelta.x + relativeVelocity.y * normalizedDelta.y) * constants.damping;
		b.y = normalizedDelta.y * (relativeVelocity.x * normalizedDelta.x + relativeVelocity.y * normalizedDelta.y) * constants.damping;
		force.x = a.x + b.x;
		force.y = a.y + b.y;
		forces[spring.indices.x].x += force.x;
		forces[spring.indices.x].y += force.y;
		forces[spring.indices.y].x -= force.x;
		forces[spring.indices.y].y -= force.y;
	}
	for (int i = 0; i < vertices.size(); i++) {
		vertices[i].velocity.x += forces[i].x / constants.mass * constants.delta_time;
		vertices[i].velocity.y += forces[i].y / constants.mass * constants.delta_time;
		vertices[i].position.x += vertices[i].velocity.x * constants.delta_time;
		vertices[i].position.y += vertices[i].velocity.y * constants.delta_time;
	}
	forces.clear();
	for (int i = 0; i < constants.vertices_size; i++) {
		if (vertices[i].position.x < 0.0f) {
			vertices[i].velocity.x *= -1;
			vertices[i].position.x = 0.0001;
		}
		if (vertices[i].position.y < 0.0f) {
			vertices[i].velocity.y *= -1;
			vertices[i].position.y = 0.0001;
		}
		if (vertices[i].position.x > 1.0f) {
			vertices[i].velocity.x *= -1;
			vertices[i].position.x = 0.9990;
		}
		if (vertices[i].position.y > 1.0f) {
			vertices[i].velocity.y *= -1;
			vertices[i].position.y = 0.9999;
		}
	}
	updateMesh();
}
