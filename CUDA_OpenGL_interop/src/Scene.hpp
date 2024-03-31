#pragma once
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "SoftBody.cuh"
#include "ParticleSystem.cuh"
#include "Properties.hpp"
#include "Camera.hpp"
#include "Shader.hpp"

class Scene {
public:
	Scene() {}
	Scene(std::vector<SoftBody> softBodies, std::vector<Shader> shaders);
	void load(const std::string& project_path);
	void render();
	void setCameraZoom(const OrthographicCamera& camera, FrameHandler& input);
	void setCameraProjection(const OrthographicCamera& camera);
	void updateShaders(OrthographicCamera& camera);
	void updateMeshes();
private:
	void processInput();
	void simulateMeshes();
	void loadSoftBodies();
	void loadParticles();
	void loadShaders();
	std::string scene_path;
	std::vector<SoftBody> softBodies;
	std::vector<ParticleSystem> particles;
	std::vector<Shader> shaders;
};