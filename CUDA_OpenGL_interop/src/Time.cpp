#include "Time.hpp"

float Time::delta_time = 0.0f;
float Time::last_frame_time = 0.0f;
float Time::current_time = 0.0f;
float Time::frames_per_second = 0.0f;

void Time::update_time()
{
    current_time = static_cast<float>(glfwGetTime());
    delta_time = current_time - last_frame_time;
    last_frame_time = current_time;
    frames_per_second = 1.0f/delta_time;
}

