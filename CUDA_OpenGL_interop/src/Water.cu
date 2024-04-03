#include "Water.cuh"

void FlipFluid::init(std::string&& shaders_path)
{
	shader = Shader(shaders_path + "/canvas.vert", shaders_path + "/canvas.frag");
	cell_size = 0.02;
	particle_radius = 0.005;
	if (boundings.x > boundings.z) {
		std::swap(boundings.x, boundings.z);
	}
	if (boundings.y > boundings.w) {
		std::swap(boundings.y, boundings.w);
	}
	size.x = boundings.z - boundings.x;
	size.y = boundings.w - boundings.y;
	resolution = size / cell_size;


	float2* h_solid_cells = new float2[resolution.x * resolution.y];
	for (int y = 0; y < resolution.y; y++) {
		for (int x = 0; x < resolution.x; x++) {
			float2 val = make_float2(0, 0);
			val.x = x < 1 || x > resolution.x - 1 ? 0 : 1;
			val.y = y < 1 || y > resolution.y - 1 ? 0 : 1;
			h_solid_cells[x * (int)resolution.y + y] = val;
		}
	}

	glGenTextures(1, &id_solid_cells);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id_solid_cells);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, resolution.x, resolution.y, 0, GL_RG, GL_FLOAT, nullptr);
	glClearTexImage(id_solid_cells, 0, GL_RG, GL_FLOAT, &glm::vec2(0.1)[0]);
	glBindImageTexture(0, id_solid_cells, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id_solid_cells);

	//velocity = Surface<float2>(make_float2(resolution.x, resolution.y));
	//solid_cells = Surface<float2>(make_float2(resolution.x, resolution.y));
	//grid = Surface<float>(make_float2(resolution.x, resolution.y));
	//density = Surface<float>(make_float2(resolution.x, resolution.y));
	//sum_of_weights = Surface<float>(make_float2(resolution.x, resolution.y));

	//particles_size = size.x * size.y * rest_particle_density;
	//particles = std::vector<Particle>(particles_size);
	//std::uniform_real_distribution<float> rand_width(boundings.x + cell_size, boundings.z - cell_size);
	//std::uniform_real_distribution<float> rand_height(boundings.y + cell_size, boundings.w - cell_size);
	//for (int i = 0; i < particles_size; i++) {
	//	particles[i].position = make_float2(rand_width(gen), rand_height(gen));
	//	particles[i].velocity = make_float2(0, 0);
	//}

	//glGenVertexArrays(1, &VAO);
	//glGenBuffers(1, &VBO);
	//glBindVertexArray(VAO);

	//glBindBuffer(GL_ARRAY_BUFFER, VBO);
	//glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), &particles[0], GL_DYNAMIC_DRAW);

	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, position));

	//glEnableVertexAttribArray(1);
	//glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, velocity));

	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	//glBindVertexArray(0);

	//size_t num_bytes;
	//cudaGraphicsResource* cuda_vbo_resource;
	//cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsNone);
	//cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	//cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, cuda_vbo_resource);
	//cudaGraphicsUnmapResources(1, &cuda_vbo_resource);
	//collisions.setup(size.x, size.y, particles.size(), particle_radius, d_particles);
}

void FlipFluid::update()
{
	
}

void FlipFluid::draw()
{
	shader.use();
	unsigned int quadvao = 0, quadvbo = 0;
	if (quadvao == 0) {
		float quadVertices[] = {
			// positions  // texture Coords
			-1.0f,  1.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadvao);
		glGenBuffers(1, &quadvbo);
		glBindVertexArray(quadvao);
		glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	}
	glBindTexture(GL_TEXTURE_2D, id_solid_cells);
	glBindVertexArray(quadvao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}
