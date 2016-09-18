#version 330 core

layout(location = 0) in vec4 in_Position;
layout(location = 1) in vec2 in_Texcoord;
layout(location = 2) in vec3 in_Normal;

smooth out vec2 texcoord;

void main()
{
  texcoord = in_Texcoord;   
  gl_Position = in_Position;
}