#pragma once


#include <glm/glm.hpp>

#include <math.h>
#include "FileLoaders.hpp"

class Properties {
public:
	//Properties() {}
	Properties() : scr_width(800), scr_height(600), fullscreen(false), background_color(glm::vec3(0, 0, 0)), frame_rate_limit(60) { name = "";}
	void load(const std::string& project_path);
	std::string name;
	uint32_t scr_width;
	uint32_t scr_height;
	bool fullscreen;
	glm::vec3 background_color;
	uint32_t frame_rate_limit;
};