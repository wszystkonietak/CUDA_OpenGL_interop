

#version 460 core
out vec4 FragColor;


void main()
{
	float dist = distance(vec2(0.5, 0.5), gl_PointCoord.xy);
	if (dist > 0.5)
        discard;
	FragColor = vec4(1.0f, 0, 0, 1.0);
}
