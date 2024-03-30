#include "Properties.hpp"

void Properties::load(const std::string& project_path)
{
	std::vector<std::string> lines = loadFile(project_path + "/properties.txt");
	for (auto& line : lines) {
		size_t delimiter_pos = line.find(": ");
		if (delimiter_pos != std::string::npos) {
			std::string key = line.substr(0, delimiter_pos);
			std::string value = line.substr(delimiter_pos + 2);

			if (key == "project name") {
				name = value;
			}
			else if (key == "width") {
				scr_width = std::stoi(value);
			}
			else if (key == "height") {
				scr_height = std::stoi(value);
			}
			else if (key == "fullscreen") {
				fullscreen = (value == "true");
			}
			else if (key == "background color") {
				std::vector<std::string> color = split(value, ", ");
				background_color = glm::vec3(std::stof(color[0]), std::stof(color[1]), std::stof(color[2]));
			}
			else if (key == "frame rate limit") {
				frame_rate_limit = std::stoi(value);
			}
		}
	}
}
