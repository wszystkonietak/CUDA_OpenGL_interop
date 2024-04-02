#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Project.hpp"
#include "Callbacks.hpp"

int main()
{
	std::string path = "C:/Users/Igor/Desktop/myProject/CUDA_OpenGL_interop/CUDA_OpenGL_interop/wszystkonietak";
	Project project(path);
	Callbacks callbacks(project);
	project.run();
	return EXIT_SUCCESS;
}