#version 330
in vec3 normal;
in float depth;
out vec4 oColor;
out vec4 FragColor;

void main(){
    // FragColor = vec4(normalize(normal), depth) * 0.5 +0.5;
    //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    oColor = vec4(normalize(normal), depth) * 0.5 +0.5;
}