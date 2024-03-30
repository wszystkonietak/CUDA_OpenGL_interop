#pragma once

#include "Project.hpp"
#include "FrameHandler.hpp"
#include "FunctionWrapper.hpp"


class Callbacks {
public:
	Callbacks(Project& project);
	static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
private:
	void setInputResponse(MouseHandler&& mouse, Function<void(FrameHandler& input)>&& func);
	//void (Callbacks::*mouse_responses[mouse_responses_size])();
	std::vector<Function<void(FrameHandler& input)>> responses[mouse_responses_size];
	Project& project;
	Properties properties;
	FrameHandler input;
};