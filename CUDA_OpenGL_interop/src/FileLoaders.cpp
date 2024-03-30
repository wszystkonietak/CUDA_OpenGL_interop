#include "FileLoaders.hpp"

std::vector<std::string> loadFile(const std::string& filename)
{
	std::vector<std::string> lines;
	std::ifstream file(filename);

	std::string line;
	while (std::getline(file, line)) {
		lines.push_back(line);
	}

	file.close();
	return lines;
}

//std::vector<std::string> split(const std::string& str, char delimiter)
//{
//	std::vector<std::string> result;
//	std::stringstream ss{ str };
//	std::string item;
//	while (std::getline(ss, item, delimiter)) {
//		result.push_back(item);
//	}
//	return result;
//}

std::vector<std::string> split(std::string& str, std::string delimeter)
{
	std::vector<std::string> res;
	std::string token = "";
	for (int i = 0; i < str.size(); i++) {
		bool flag = true;
		for (int j = 0; j < delimeter.size(); j++) {
			if (str[i + j] != delimeter[j]) flag = false;
		}
		if (flag) {
			if (token.size() > 0) {
				res.push_back(token);
				token = "";
				i += delimeter.size() - 1;
			}
		}
		else {
			token += str[i];
		}
	}
	res.push_back(token);
	return res;
}

//void loadScene(Scene& scene, const std::string& project_path)
//{
//	std::string scene_path = project_path + "/Scene";
//	std::vector<SoftBody> softBodies;
//	std::vector<Shader> shaders;
//
//	loadSoftBodies(softBodies, scene_path);
//
//	loadShaders(shaders, scene_path);
//
//	scene.shaders = shaders;
//	scene.softBodies = softBodies;
//}

//void loadProperties(Properties& properties, const std::string& project_path)
//{
//	std::vector<std::string> lines = loadFile(project_path + "/properties.txt");
//	for (auto& line : lines) {
//		size_t delimiterPos = line.find(": ");
//		if (delimiterPos != std::string::npos) {
//			std::string key = line.substr(0, delimiterPos);
//			std::string value = line.substr(delimiterPos + 2);
//
//			if (key == "project name") {
//				properties.name = value;
//			}
//			else if (key == "width") {
//				properties.scr_width = std::stoi(value);
//			}
//			else if (key == "height") {
//				properties.scr_height = std::stoi(value);
//			}
//			else if (key == "fullscreen") {
//				properties.fullscreen = (value == "true");
//			}
//			else if (key == "background color") {
//				std::vector<std::string> color = split(value, ", ");
//				properties.background_color = glm::vec3(std::stof(color[0]), std::stof(color[1]), std::stof(color[2]));
//			}
//			else if (key == "frame rate limit") {
//				properties.frame_rate_limit = std::stoi(value);
//			}
//		}
//	}
//}

//void loadCamera(OrthographicCamera& camera, const std::string& project_path)
//{
//	std::string orthographicCameraPath = project_path + "/Camera/orthographicCamera.txt";
//	std::vector<std::string> lines = loadFile(orthographicCameraPath);
//	float_t right = 0, left = 0, top = 0, bottom = 0, near = 0, far = 0;
//	glm::vec3 position(0, 0, 0);
//	for (auto& line : lines) {
//		size_t delimiterPos = line.find(": ");
//		if (delimiterPos != std::string::npos) {
//			std::string key = line.substr(0, delimiterPos);
//			std::string value = line.substr(delimiterPos + 2);
//			if (key == "right") {
//				right = std::stof(value);
//			}
//			else if (key == "left") {
//				left = std::stof(value);
//			}
//			else if (key == "top") {
//				top = std::stof(value);
//			}
//			else if (key == "bottom") {
//				bottom = std::stof(value);
//			}
//			else if (key == "near") {
//				near = std::stof(value);
//			}
//			else if (key == "far") {
//				far = std::stof(value);
//			}
//			else if (key == "position") {
//				std::vector<std::string> s_position = split(value, ", ");
//				position = glm::vec3(std::stof(s_position[0]), std::stof(s_position[1]), std::stof(s_position[2]));
//			}
//		}
//	}
//	camera.init(left, right, bottom, top, near, far, position);
//}

//void loadSoftBodies(std::vector<SoftBody>& softBodies, std::string& project_path)
//{
//	std::vector<std::string> lines = loadFile(project_path + "/softBodies.txt");
//	lines.erase(lines.begin());
//	for (auto& line : lines) {
//		std::vector<std::string> numbers = split(line, ", ");
//		if (numbers.size()) {
//			float_t sizeX = std::stof(numbers[0]);
//			float_t sizeY = std::stof(numbers[1]);
//			float_t restLength = std::stof(numbers[2]);
//			float_t startX = std::stof(numbers[3]);
//			float_t startY = std::stof(numbers[4]);
//
//			softBodies.push_back(SoftBody(glm::vec2(sizeX, sizeY), restLength, glm::vec2(startX, startY)));
//		}
//	}
//}
//
//void loadShaders(std::vector<Shader>& shaders, const std::string& project_path) 
//{
//	std::string shaderFolderPath = project_path + "/Shaders/";
//	std::vector<std::string> lines = loadFile(shaderFolderPath + "shaders.txt");
//	for (auto& line : lines) {
//		std::ifstream file(line + ".geom");
//		if (file.is_open()) {
//			shaders.push_back(Shader(shaderFolderPath + line + ".vert", shaderFolderPath + line + ".frag", shaderFolderPath + line + ".geom"));
//			file.close();
//		}
//		else {
//			shaders.push_back(Shader(shaderFolderPath + line + ".vert", shaderFolderPath + line + ".frag", ""));
//		}
//	}
//}