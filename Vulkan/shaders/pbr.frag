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
layout(binding = 7) uniform samplerCube irradianceCube;


layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;


layout(location = 0) out vec4 outColor;

#define PI 3.1415926535897932384626433832795
#define ALBEDO pow(texture(albedo, fragTexCoord).rgb, vec3(2.2))
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

vec3 Uncharted2Tonemap(vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom); 
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
vec3 F_SchlickR(float cosTheta, vec3 F0, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

// vec3 prefilteredReflection(vec3 R, float roughness)
// {
// 	const float MAX_REFLECTION_LOD = 9.0; // todo: param/const
// 	float lod = roughness * MAX_REFLECTION_LOD;
// 	float lodf = floor(lod);
// 	float lodc = ceil(lod);
// 	vec3 a = textureLod(prefilteredMap, R, lodf).rgb;
// 	vec3 b = textureLod(prefilteredMap, R, lodc).rgb;
// 	return mix(a, b, lod - lodf);
// }

vec3 specularContribution(vec3 L, vec3 V, vec3 N, vec3 F0, float metallic, float roughness)
{
	// Precalculate vectors and dot products	
	vec3 H = normalize (V + L);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);

	// Light color fixed
	vec3 lightColor = vec3(1.0);

	vec3 color = vec3(0.0);

	if (dotNL > 0.0) {
		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness); 
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		vec3 F = F_Schlick(dotNV, F0);		
		vec3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);		
		vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);			
		color += (kD * ALBEDO / PI + spec) * dotNL;
	}

	return color;
}
void main() {
    
    //outColor = texture(albedo, fragTexCoord);
    vec3 N = perturbNormal();
    vec3 V = normalize(matrices.viewPos - fragPos);
    vec3 sampleColor = ALBEDO;
    float aoValue = texture(ao, fragTexCoord).r;
    float metallicValue = texture(metallic, fragTexCoord).r;
    float roughnessValue = texture(metallic, fragTexCoord).r;
    vec3 F0 = vec3(0.04f);
    F0 = mix(F0, sampleColor, roughnessValue);
    vec3 Lo = vec3(0.0f);
    for(int i = 0; i < 4; i++) {
        vec3 L = normalize(params.positions[i].xyz - fragPos);
        Lo += specularContribution(L, V, N, F0, metallicValue, roughnessValue);

    }
	vec3 irradiance = texture(irradianceCube, N).rgb;
	// Diffuse based on irradiance
	vec3 diffuse = irradiance * sampleColor;	
	vec3 F = F_SchlickR(max(dot(N, V), 0.0), F0, roughnessValue);
	// Ambient part
	vec3 kD = 1.0 - F;
	kD *= 1.0 - metallicValue;
    vec3 ambient = kD * diffuse * sampleColor * aoValue;
    vec3 color   = ambient + Lo;
    color = Uncharted2Tonemap(color * params.exposure);
	color = color * (1.0f / Uncharted2Tonemap(vec3(11.2f)));
    // color = color / (vec3(1.0) + color);
    outColor = vec4(pow(color, vec3(1.0f / params.gamma)), 1.0);
    // outColor = texture(texSampler, fragTexCoord) * vec4(light.color, 1.0);
}
