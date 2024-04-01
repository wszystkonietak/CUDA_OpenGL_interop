#include "Callbacks.hpp"

Callbacks::Callbacks(Project& project) : project(project), input(project.properties)
{
	properties = project;
	/*responses[MouseHandler(scroll_up)].push_back([&](FrameHandler& input) {project.camera.zoomIn(input); });*/
	glfwSetFramebufferSizeCallback(project, framebufferSizeCallback);
	glfwSetKeyCallback(project, keyCallback);
	glfwSetCursorPosCallback(project, cursorPositionCallback);
	glfwSetMouseButtonCallback(project, mouseButtonCallback);
	glfwSetScrollCallback(project, scrollCallback);
	glfwSetWindowUserPointer(project, this);

	setInputResponse(MouseHandler(Scroll(scroll_up)), [&project](FrameHandler& input) 
	{
		project.camera.zoomIn(input); 
		project.scene.setCameraZoom(project.camera, input);
	});
	setInputResponse(MouseHandler(Scroll(scroll_down)), [&project](FrameHandler& input)
	{
		project.camera.zoomOut(input); 
		project.scene.setCameraZoom(project.camera, input);
	});
	setInputResponse(MouseHandler(Buttons(left_button, held), Cursor(is_moving)), [&project](FrameHandler& input)
	{
		project.camera.updatePosition(input);
		project.scene.setCameraProjection(project.camera); 
	});
	setInputResponse(MouseHandler(Buttons(right_button, held), Cursor(is_moving)), [&project](FrameHandler& input)
	{
			project.camera.updatePosition(input);
			project.scene.setCameraProjection(project.camera);
	});
}

void Callbacks::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void Callbacks::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void Callbacks::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
{
	Callbacks* state = static_cast<Callbacks*>(glfwGetWindowUserPointer(window));
	ypos = (state->properties.scr_height - ypos) / (double)state->properties.scr_height;
	xpos /= (double)state->properties.scr_width;
	state->input.mouse.setCursorPosition(xpos, ypos);
	for (auto& func : state->responses[state->input.mouse.id])
		func(state->input);
	//(state->*state->mouse_responses[state->input.mouse])(state->input);
	/*InputHandler::mouse.process_cursor_position(xpos, ypos);
	if (InputHandler::mouse.is_clicked[GLFW_MOUSE_BUTTON_LEFT]) {
		state->camera.updatePosition();
		state->scene.setCameraProjection(state->camera);
	}*/
}

void Callbacks::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	Callbacks* state = static_cast<Callbacks*>(glfwGetWindowUserPointer(window));
	state->input.mouse.buttons.set(button, action);
	for (auto& func : state->responses[state->input.mouse.buttons.id])
		func(state->input);
}

void Callbacks::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	if (!yoffset)
		return;
	Callbacks* state = static_cast<Callbacks*>(glfwGetWindowUserPointer(window));
	state->input.mouse.scroll.set(yoffset);
	for (auto& func : state->responses[state->input.mouse.scroll.id])
		func(state->input);
}

void Callbacks::setInputResponse(MouseHandler&& mouse, Function<void(FrameHandler& input)>&& func)
{
	responses[mouse].push_back(std::move(func));
}
