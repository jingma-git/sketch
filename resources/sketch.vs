#version 330
layout (location=0) in vec3 aPosition;
layout (location=1) in vec3 aNormal;

uniform mat4 uViewMatrix;
uniform mat4 uRotateMatrix;

out vec3 normal;
out float depth;

void main(){
    gl_Position = uViewMatrix * vec4(aPosition, 1.0);
    normal = (uRotateMatrix * vec4(aNormal, 1.0)).xyz;
    if(normal.z<0.0) normal = -normal;
    depth = gl_Position.z;
    // gl_Position = vec4(aPosition, 1.0);
}