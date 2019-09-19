#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform Matrices {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 viewPos;
} matrices;
layout (binding = 1) uniform Params {
    vec4[4] positions;
    float exposure;
    float gamma;
} params;
layout(binding = 2) uniform sampler2D albedo;
layout(binding = 3) uniform sampler2D normal;


layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;


layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(albedo, fragTexCoord);
    // outColor = texture(texSampler, fragTexCoord) * vec4(light.color, 1.0);
}
