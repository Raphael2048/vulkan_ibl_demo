#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 viewPos;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
} ubo;
// layout(binding = 1) uniform DirectLight {
//     vec3 direction;
//     vec3 ambient;
//     vec3 diffuse;
//     vec3 specular;
// } dirLight;
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;


layout(location = 0) out vec4 outColor;

void main() {
    vec3 viewDir = normalize(ubo.viewPos - fragPos);
    vec3 lightDir = normalize(-ubo.direction);
    vec3 sampleColor = vec3(texture(texSampler, fragTexCoord));

    float diff = max(dot(fragNormal, lightDir), 0.0);
    //vec3 diffuse = ubo.diffuse * diff * sampleColor;
    vec3 diffuse = diff * sampleColor;

    vec3 ambient = ubo.ambient * sampleColor;

    vec3 reflectDir = reflect(-lightDir, fragNormal);
    float spec = pow(max(dot(reflectDir, viewDir), 0.0), 5);
    vec3 specular = ubo.specular * spec * sampleColor;
    outColor = vec4(diffuse + ambient + specular, 1.0);
    //outColor = vec4(diffuse, 1.0);
    //outColor = texture(texSampler, fragTexCoord);
}
