#include "Scene.hpp"

Scene::Scene(std::vector<SoftBody> softBodies, std::vector<Shader> shaders)
{
	this->softBodies = softBodies;
	this->shaders = shaders;
}

void Scene::load(const std::string& project_path)
{
	scene_path = project_path + "/Scene";
	
	loadSoftBodies();
	loadParticles();
	loadShaders();
	loadCanvases();
}

void Scene::updateShaders(OrthographicCamera& camera)
{
	shaders[s_Basic].use();
	shaders[s_Basic].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	shaders[s_SoftBody].use();
	shaders[s_SoftBody].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	shaders[s_SoftBody].use();
	shaders[s_SoftBody].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
}

void Scene::updateMeshes()
{
	for (auto& softBody : softBodies) {
		softBody.simulate();
	}
	for (auto& particleSystem : particles) {
		particleSystem.update();
	}
	for (auto& canvas : canvases) {
		canvas.update();
	}
}

void Scene::render()
{
	shaders[s_Basic].use();
	for (auto& softBody : softBodies) {
		softBody.draw(shaders[s_SoftBody], points);
	}
	for (auto& particleSystem : particles) {
		particleSystem.draw(shaders[s_Particles]);
	}
	for (auto& canvas : canvases) {
		canvas.draw();
	}
}

void Scene::setCameraZoom(const OrthographicCamera& camera, FrameHandler& input)
{
	shaders[s_Basic].use();
	shaders[s_Basic].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	float inPixelDiameter = 0;
	if (softBodies.size() > 0) {
		inPixelDiameter = 2* softBodies[0].particle_radius * camera.getFrustumRatio();
		shaders[s_Basic].setFloat("u_inPixelDiameter", inPixelDiameter);
	}

	shaders[s_SoftBody].use();
	shaders[s_SoftBody].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	if (softBodies.size() > 0) {
		shaders[s_SoftBody].setFloat("u_inPixelDiameter", inPixelDiameter);
	}

	shaders[s_Particles].use();
	shaders[s_Particles].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	if (particles.size() > 0) {
		float inPixelRadius = particles[0].particle_radius * (input.properties.scr_width / camera.getFrustum().y);
		inPixelDiameter = 2 * inPixelRadius;
		shaders[s_Particles].setFloat("u_inPixelRadius", inPixelRadius);
		shaders[s_Particles].setFloat("u_inPixelDiameter", inPixelDiameter);
	}
	
}

void Scene::setCameraProjection(const OrthographicCamera& camera)
{
	shaders[s_Basic].use();
	shaders[s_Basic].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());

	shaders[s_SoftBody].use();
	shaders[s_SoftBody].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());

	shaders[s_Particles].use();
	shaders[s_Particles].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
}

void Scene::loadSoftBodies()
{
	std::vector<std::string> lines = loadFile(scene_path + "/softBodies.txt");
	lines.erase(lines.begin());
	for (auto& line : lines) {
		std::vector<std::string> numbers = split(line, ", ");
		if (numbers.size()) {
			float_t sizeX = std::stof(numbers[0]);
			float_t sizeY = std::stof(numbers[1]);
			float_t restLength = std::stof(numbers[2]);
			float_t startX = std::stof(numbers[3]);
			float_t startY = std::stof(numbers[4]);

			softBodies.push_back(SoftBody(glm::vec2(sizeX, sizeY), restLength, glm::vec2(startX, startY)));
		}
	}
}

void Scene::loadParticles()
{
	std::vector<std::string> lines = loadFile(scene_path + "/particles.txt");
	lines.erase(lines.begin());
	for (auto& line : lines) {
		std::vector<std::string> numbers = split(line, ", ");
		if (numbers.size()) {
			float_t scene_width = std::stof(numbers[0]);
			float_t scene_height = std::stof(numbers[1]);
			float_t particle_radius = std::stof(numbers[2]);
			int num_particles = std::stoi(numbers[3]);

			particles.push_back(ParticleSystem(scene_width, scene_height, particle_radius, num_particles));
		}
	}
}

void Scene::loadShaders()
{
	std::string shaderFolderPath = scene_path + "/Shaders/";
	std::vector<std::string> lines = loadFile(shaderFolderPath + "shaders.txt");
	for (auto& line : lines) {
		std::ifstream file(line + ".geom");
		if (file.is_open()) {
			shaders.push_back(Shader(shaderFolderPath + line + ".vert", shaderFolderPath + line + ".frag", shaderFolderPath + line + ".geom"));
			file.close();
		}
		else {
			shaders.push_back(Shader(shaderFolderPath + line + ".vert", shaderFolderPath + line + ".frag", ""));
		}
	}
}

void Scene::loadCanvases()
{
	canvases.push_back(Canvas(glm::vec4(-1, -1, 1, 1), glm::vec2(100, 100), scene_path + "/Shaders/"));
}
