#pragma once

#include "MouseHandler.hpp"


class FrameHandler {
public:
	FrameHandler(Properties& props) : properties(props) {  };
	MouseHandler mouse;
	const Properties& properties;
};

