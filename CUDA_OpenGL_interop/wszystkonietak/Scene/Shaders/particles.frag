

#version 460 core
out vec4 FragColor;

uniform float u_inPixelRadius;
in flat int VertexID; 

void main()
{
	float dist = distance(vec2(0.5, 0.5), gl_PointCoord.xy);
	if (dist > 0.5)
        discard;
	FragColor = vec4(.5,1,1, 1.0);
}

