#version 330 core
layout (location=0) in vec3 aPosition;
layout (location=1) in vec3 aNormal;


uniform mat4 uViewMatrix;
uniform mat4 uRotateMatrix;

out vec3 normal;

void main()
{
    normal = aNormal;
    gl_Position = vec4(aPosition.x, aPosition.y, aPosition.z, 1.0);
}