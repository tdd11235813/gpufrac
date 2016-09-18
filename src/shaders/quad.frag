#version 330 core

uniform sampler2D sampler0;

smooth in vec2 texcoord;

out vec4 out_Color; 

void main()
{	
  vec4 color = texture2D(sampler0, texcoord);
  out_Color = color;
}