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
layout(binding = 4) uniform sampler2D ao;
layout(binding = 5) uniform sampler2D metallic;
layout(binding = 6) uniform sampler2D roughness;


layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;


layout(location = 0) out vec4 outColor;


vec3 perturbNormal()
{
	vec3 tangentNormal = texture(normal, fragTexCoord).xyz * 2.0 - 1.0;

	vec3 q1 = dFdx(fragPos);
	vec3 q2 = dFdy(fragPos);
	vec2 st1 = dFdx(fragTexCoord);
	vec2 st2 = dFdy(fragTexCoord);

	vec3 N = normalize(fragNormal);
	vec3 T = normalize(q1 * st2.t - q2 * st1.t);
	vec3 B = -normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	return normalize(TBN * tangentNormal);
}
void main() {
    //outColor = texture(albedo, fragTexCoord);
    vec4 sampleColor = texture(albedo, fragTexCoord);
    vec3 tnormal = perturbNormal();
    float k = 0;
    for (int i = 0 ; i < 4; i++) {
        vec3 viewDir = normalize(matrices.viewPos - fragPos);
        vec3 lightDir = normalize(vec3(params.positions[i]) - fragPos);
        float diffuse = max(dot(lightDir, tnormal), 0.0);
        vec3 halfDir = normalize(viewDir + lightDir);
        float spec = pow(max(dot(halfDir, tnormal), 0.0), 5);
        k += diffuse * 0.2f + spec * 0.2f;
    }
    outColor = pow((k + 0.1f) * sampleColor, vec4(1.0f / params.gamma));
    // outColor = texture(texSampler, fragTexCoord) * vec4(light.color, 1.0);
}
