#include "Scene.hpp"

Scene::Scene(std::vector<SoftBody> softBodies, std::vector<Shader> shaders)
{
	this->softBodies = softBodies;
	this->shaders = shaders;
}

void Scene::load(const std::string& project_path)
{
	std::string scene_path = project_path + "/Scene";
	
	loadSoftBodies(scene_path);
	
	loadShaders(scene_path);
}

void Scene::updateShaders(OrthographicCamera& camera)
{
	shaders[s_Basic].use();
	shaders[s_Basic].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	shaders[s_SoftBody].use();
	shaders[s_SoftBody].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
}

void Scene::updateMeshes()
{
	processInput();
	simulateMeshes();

}

void Scene::render()
{
	shaders[s_Basic].use();
	for (auto& softBody : softBodies) {
		softBody.draw(shaders[s_SoftBody], points);
	}
}

void Scene::setCameraZoom(const OrthographicCamera& camera)
{
	shaders[s_Basic].use();
	shaders[s_Basic].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	float inPixelDiameter = 2* softBodies[0].particle_radius * camera.getFrustumRatio();
	shaders[s_Basic].setFloat("u_inPixelDiameter", inPixelDiameter);

	shaders[s_SoftBody].use();
	shaders[s_SoftBody].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
	shaders[s_SoftBody].setFloat("u_inPixelDiameter", inPixelDiameter);
}

void Scene::setCameraProjection(const OrthographicCamera& camera)
{
	shaders[s_Basic].use();
	shaders[s_Basic].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());

	shaders[s_SoftBody].use();
	shaders[s_SoftBody].setMat4("u_projectionViewMatrix", camera.getProjectionViewMatrix());
}

void Scene::processInput()
{
	for (auto& softBody : softBodies) {
		/*if (InputHandler::mouse.is_double_clicked[GLFW_MOUSE_BUTTON_LEFT] && InputHandler::mouse.double_click_cout[GLFW_MOUSE_BUTTON_LEFT] % 2 == 1) {
			softBody.update_sticked_particle_id();
		}*/
	}
}

void Scene::simulateMeshes()
{
	for (auto& softBody : softBodies) {
		softBody.simulate();
	}
}

void Scene::loadSoftBodies(const std::string& project_path)
{
	std::vector<std::string> lines = loadFile(project_path + "/softBodies.txt");
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

void Scene::loadShaders(const std::string& project_path)
{
	std::string shaderFolderPath = project_path + "/Shaders/";
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