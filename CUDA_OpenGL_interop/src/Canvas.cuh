#include "DataTypes.hpp"
#include "Shader.hpp"

struct CanvasConstants {
	float time;
	float2 texture_resolution;
};

typedef float4 (*UpdateFunctionPtr)(float2, CanvasConstants*);

class Canvas {
public:
	Canvas() = default;
	Canvas(glm::vec4&& boundings, glm::vec2&& texture_size, std::string&& shaders_path) { init(std::move(boundings), std::move(texture_size), std::move(shaders_path)); };
	void setUpdateFunction(UpdateFunctionPtr d_func) { this->update_function = d_func; };
	void init(glm::vec4&& boundings, glm::vec2&& texture_size, std::string&& shaders_path);
	void update();
	void draw();
	UpdateFunctionPtr update_function;
	glm::vec4 boundings;
	glm::vec2 texture_size;
	GLuint texture;
	Shader shader;
	GLuint VAO, VBO;
};


