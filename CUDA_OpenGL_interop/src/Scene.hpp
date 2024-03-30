#pragma once
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "SoftBody.cuh"
#include "Properties.hpp"
#include "Camera.hpp"
#include "Shader.hpp"

class Scene {
public:
	Scene() {}
	Scene(std::vector<SoftBody> softBodies, std::vector<Shader> shaders);
	void load(const std::string& project_path);
	void render();
	void setCameraZoom(const OrthographicCamera& camera);
	void setCameraProjection(const OrthographicCamera& camera);
	void updateShaders(OrthographicCamera& camera);
	void updateMeshes();
private:
	void processInput();
	void simulateMeshes();
	void loadSoftBodies(const std::string& project_path);
	void loadShaders(const std::string& project_path);
	std::vector<SoftBody> softBodies;
	std::vector<Shader> shaders;
};