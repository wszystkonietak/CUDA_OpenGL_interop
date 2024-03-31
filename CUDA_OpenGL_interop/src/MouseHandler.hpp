#pragma once

#include <glm/glm.hpp>
#include <chrono>
#include "DataTypes.hpp"
#include "Time.hpp"
#include "Properties.hpp"

enum {
	bits_for_id = 5,
	bits_for_scroll = 2,
	bits_for_button = 4,
	mouse_responses_size = 64,
	scroll_up = 0b11,
	scroll_down = 0b01,
	left_button = GLFW_MOUSE_BUTTON_LEFT,
	right_button = GLFW_MOUSE_BUTTON_RIGHT,
	middle_button = GLFW_MOUSE_BUTTON_MIDDLE,
	release = GLFW_RELEASE,
	press = GLFW_PRESS,
	held = 3,
	is_moving = 1,
};

struct Cursor {
	Cursor()
		: id(1 << bits_for_button), current(0)
	{}
	Cursor(int a) {
		id = 1 << bits_for_button;
	}
	void setPosition(double xpos, double ypos) {
		this->previous = this->current;
		this->current = glm::vec2(xpos, ypos);
		this->delta.x = xpos - this->previous.x;
		this->delta.y = ypos - this->previous.y;
	}
	int id;
	glm::vec2 current;
	glm::vec2 previous;
	glm::vec2 delta;
	glm::vec2 scene;
};

struct Buttons {
	Buttons() {
		pressed_count = 0;
	}
	Buttons(int button, int action) {
		id = action == held ? button + 1 : ((button << 1) | action) << 2;
	}
	void set(int button, int action) {
		break_duration[button] = Time::current_time - last_click[button];
		last_click[button] = Time::current_time;
		count[button]++;
		is_clicked[button] = action;
		if (action == press) {
			pressed_count++;
			last_pressed_id = button + 1;
		}
		else
			pressed_count--;
		//pressed_count += action == press ? 1 : -1;
		id = ((button << 1) | action) << 2;
	}
	int id;
	int last_pressed_id;
	int pressed_count;
	bool is_clicked[8];
	int count[8];
	float last_click[8];
	float break_duration[8];
};

struct Scroll {
	Scroll() {}
	Scroll(int direction) { set(direction); }
	Scroll(double offset) { set(offset); }
	void set(int direction) {
		is_up = direction == scroll_up ? true : false;
		id = direction;
	}
	void set(double offset) {
		this->is_up = offset > 0 ? true : false;
		id = offset > 0 ? scroll_up : scroll_down;
	}
	int id;
	bool is_up;
};

class MouseHandler {
public:
	MouseHandler() : cursor(), buttons(), scroll() {}
	MouseHandler(Scroll&& scroll) : id{ scroll.id } {}
	MouseHandler(Buttons&& button) : id{ button.id } {}
	MouseHandler(Cursor&& cursor) : id{ cursor.id } {}
	MouseHandler(Buttons&& button, Cursor&& cursor) : id{ button.id | cursor.id } {}

	void setButton(int button, int action);
	void setCursorPosition(double xpos, double ypos) {
		cursor.setPosition(xpos, ypos);
		id = buttons.pressed_count == 1 ? cursor.id | buttons.last_pressed_id : cursor.id;
	}
	Cursor cursor;
	Buttons buttons;
	Scroll scroll;
	operator unsigned int() const { return this->id; }
	int id;
private:
};
