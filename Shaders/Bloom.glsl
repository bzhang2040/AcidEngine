#ifdef CXX_STAGE
    #define Bloom_glsl "Bloom.glsl", "BLOOM_STAGE", "graphics"
#endif

#ifdef VERTEX_STAGE
layout(location = 0) in float _;

out vec2 texcoord;

void main() {
    if (gl_VertexID == 0) gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
    if (gl_VertexID == 1) gl_Position = vec4(1.0, -1.0, 0.0, 1.0);
    if (gl_VertexID == 2) gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);
    if (gl_VertexID == 3) gl_Position = vec4(1.0, -1.0, 0.0, 1.0);
    if (gl_VertexID == 4) gl_Position = vec4(1.0, 1.0, 0.0, 1.0);
    if (gl_VertexID == 5) gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);

    texcoord = gl_Position.xy * 0.5 + 0.5;
};
#endif



#ifdef FRAGMENT_STAGE
layout(location = 0) out vec4 OutColor;

in vec2 texcoord;

vec3 ComputeBloomTile(sampler2D sam, const float scale, vec2 offset) { // Computes a single bloom tile, the tile's blur level is inversely proportional to its size
	// Each bloom tile uses (1.0 / scale + pixelSize * 2.0) texcoord-units of the screen
	
	vec2 coord  = texcoord;
	     coord -= offset + 1 / viewSize; // A pixel is added to the offset to give the bloom tile a padding
	     coord *= scale;
	
	vec2 padding = scale / viewSize;
    
	if (any(greaterThanEqual(abs(coord - 0.5), padding + 0.5)))
		return vec3(0.0);
    
	
	float Lod = log2(scale);
	
	const float range     = 2.0 * scale; // Sample radius has to be adjusted based on the scale of the bloom tile
	const float interval  = 1.0 * scale;
	float  maxLength = length(vec2(range));
	
	vec3  bloom       = vec3(0.0);
	float totalWeight = 0.0;
	
	for (float i = -range; i <= range; i += interval) {
		for (float j = -range; j <= range; j += interval) {
			float weight  = 1.0 - length(vec2(i, j)) / maxLength;
			      weight *= weight;
			      weight  = cubesmooth(weight); // Apply a faux-gaussian falloff
			
			vec2 offset = vec2(i, j) / viewSize;
			
			vec4 lookup = textureLod(sam, clamp(coord + offset, padding/2.0, 1.0 - padding/2.0), Lod);
			
			bloom       += lookup.rgb * weight;
			totalWeight += weight;
		}
	}
	//return vec3(coord, 0.0);
	return bloom / totalWeight;
}

vec3 ComputeBloom(sampler2D sam, vec2 outOffset) {
	vec3 bloom  = ComputeBloomTile(sam,   4, vec2(0.0                          ,                           0.0) + outOffset);
	     bloom += ComputeBloomTile(sam,   8, vec2(0.0                          , 0.25     + 1/viewSize.y * 2.0) + outOffset);
	     bloom += ComputeBloomTile(sam,  16, vec2(0.125    + 1/viewSize.x * 2.0, 0.25     + 1/viewSize.y * 2.0) + outOffset);
	     bloom += ComputeBloomTile(sam,  32, vec2(0.1875   + 1/viewSize.x * 4.0, 0.25     + 1/viewSize.y * 2.0) + outOffset);
	     bloom += ComputeBloomTile(sam,  64, vec2(0.125    + 1/viewSize.x * 2.0, 0.3125   + 1/viewSize.y * 4.0) + outOffset);
	     bloom += ComputeBloomTile(sam, 128, vec2(0.140625 + 1/viewSize.x * 4.0, 0.3125   + 1/viewSize.y * 4.0) + outOffset);
	     bloom += ComputeBloomTile(sam, 256, vec2(0.125    + 1/viewSize.x * 2.0, 0.328125 + 1/viewSize.y * 6.0) + outOffset);
	
	return max(bloom, vec3(0.0));
}

void main() {
	OutColor.rgb  = ComputeBloom(Texture13, vec2(0.0, 0.0));
    OutColor.rgb += ComputeBloom(Texture9, vec2(0.5, 0.0));
};
#endif
