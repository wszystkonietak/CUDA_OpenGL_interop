#pragma once

#include "Scene.hpp"
#include "FileLoaders.hpp"
#include "Camera.hpp"
#include "Time.hpp"
#include "FrameHandler.hpp"

class Project {
public:
	Project(std::string &project_path);
	void init();
	void run();	

	OrthographicCamera camera;
	Scene scene;
	Properties properties;

	operator GLFWwindow*() const { return window; };
	operator Properties() { return properties; };
private:
	void frame_update();
	void frame_render();
	void frame_end();
	GLFWwindow* window;
	std::string project_path;
};