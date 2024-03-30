#version 460 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aVelocity;

uniform mat4 u_projectionViewMatrix;
uniform float u_inPixelDiameter;


void main()
{
	gl_PointSize = u_inPixelDiameter;
	gl_Position = u_projectionViewMatrix * vec4(aPos, 0.0, 1.0);
}