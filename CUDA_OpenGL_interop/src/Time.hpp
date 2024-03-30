#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Time {
public:
	static float delta_time;
	static float last_frame_time;
	static float current_time;
	static float frames_per_second;
	static void update_time();
};