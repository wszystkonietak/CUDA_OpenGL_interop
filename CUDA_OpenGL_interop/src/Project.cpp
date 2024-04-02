#include "Project.hpp"

Project::Project(std::string& project_path)
{
	this->project_path = project_path;
	this->init();
}

void Project::init()
{
	properties.load(project_path);
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(properties.scr_width, properties.scr_height, properties.name.c_str(), NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
	}
	glfwMakeContextCurrent(window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
	}
	glClearColor(properties.background_color.x, properties.background_color.y, properties.background_color.z, 1.0f);
	glEnable(GL_PROGRAM_POINT_SIZE);
	scene.load(project_path);
	camera.load(project_path, properties);
	scene.setCameraProjection(camera);
	scene.setCameraZoom(camera, FrameHandler(properties));
	glfwSwapInterval(0);
}

void Project::run()
{
	glfwSetTime(0.0);
	while (!glfwWindowShouldClose(window))
	{
		frame_update();
		frame_render();
		frame_end();
	}
}



void Project::frame_update()
{
	Time::update_time();
	printf("%f\n", Time::frames_per_second);
	scene.updateMeshes();
}

void Project::frame_render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	scene.render();
}

void Project::frame_end()
{
	glfwSwapBuffers(window);
	glfwPollEvents();
}
