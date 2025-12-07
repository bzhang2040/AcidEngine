
#define TOP -1
#define BOTTOM -2
#define LEFT -3
#define RIGHT -4

#define beat_marker_start -1.0f
#define beat_marker_end -2.0f

#define beat_type_default 0
#define r_low 1
#define r_default 2
#define r_wide 3
#define l_low 10
#define l_default 12
#define l_wide 13
#define b_low 21
#define b_default 22
#define b_wide 23
#define beat_type_portal 30
#define beat_type_programmatic 40
#define beat_type_nothing 50

struct BeatStructGPU {
	float beat;
	int type;
	int portalTarget;
	float zPos;
};

struct WorldRange {
	float beat;
	int zStart;
	int zEnd;
	int physicalWorldID;
	int logicalWorldID;
};

struct LogicalID {
	int id;
	int prevLogical;
	int nextLogical;
};

#define WORLD_NAME(n) (n)

#ifdef CXX_STAGE

struct BeatStruct {
	float b; // beat
	int bt = beat_type_default; // beat type
	int targetWorldName = -1;
	float d = 0.0f; // delay
};

// The main reason for this big vector is so I don't have to write a parser.
std::vector<BeatStruct> beatsArray = {
	{.b=1,.bt=beat_type_nothing},

	{.b=160,.bt=beat_type_portal,.targetWorldName=WORLD_NAME(1)},
	{.b=169,.bt=beat_type_portal,.targetWorldName=WORLD_NAME(2)},
	{.b=217,.bt=beat_type_portal,.targetWorldName=WORLD_NAME(3)},

	{.b=1078, .bt=beat_type_portal, .targetWorldName=WORLD_NAME(4)},
};

std::vector<BeatStructGPU> beatsArray2;
vector<WorldRange> portalPositions;
vector<LogicalID> physicalFromLogical(1024, {-1,-1,-1});
#endif

#undef TOP
#undef BOTTOM
#undef LEFT
#undef RIGHT

#define PI 3.14159

const int noiseTextureResolution = 64; // [16 32 64 128 256 512 1024]
const float noiseRes = float(noiseTextureResolution);
const float noiseResInverse = 1.0 / noiseRes;
const float noiseScale = 64.0 / noiseRes;

float GetBeatIndex(float beatsPerMinute, float firstTempo, float beatTime, float granularity) {
	float secondsPerBeat = 60.0 / beatsPerMinute;
	float beat = (beatTime - firstTempo) / secondsPerBeat;
	beat = round(beat / granularity) * granularity;
	return beat;
}

#define id_stone 1
#define id_grass 2
#define id_dirt  3
#define id_grass_top 4
#define id_water 9
#define id_sand  10
#define id_beat  11
#define id_oak_log 12
#define id_oak_log_top 13
#define id_leaves 14
#define id_permastone 15
#define id_null 16
#define id_stone2 17
#define id_torch 18
#define id_torch_left 19
#define id_torch_front 20
#define id_torch_right 21
#define id_torch_back 22
#define id_portal 23
#define id_portal_forward 24
#define id_portal_backward 25

bool IsPortal(uint blockID) {
	return blockID == id_portal || blockID == id_portal_forward || blockID == id_portal_backward;
}

float TorchAngle(uint blockID) {
	return float(blockID - id_torch_left);
}

bool IsTorch(uint blockID) {
	return blockID >= id_torch && blockID <= id_torch_back;
}

#define chunkUpdates 0

#define UBO_DECLARE(type, name) type name
#define UBO_DEF(type, name) UBOEntry<type> name
#define UBO_PUSH(type, name) members.push_back(&name)

#define PER_FRAME_CPU_UBO(UBO_FUNC) \
UBO_FUNC(int, shaderReload); \
UBO_FUNC(int, resetCamera); \
UBO_FUNC(int, sampleCount); \
UBO_FUNC(int, frameID); \
UBO_FUNC(float, cpu_yaw); \
UBO_FUNC(float, cpu_pitch); \
UBO_FUNC(float, cpu_zoom); \
UBO_FUNC(float, nonBlurTime); \
UBO_FUNC(float, nonBlurBeat); \
UBO_FUNC(vec3, currMovement); \
UBO_FUNC(vec3, prevRegenCameraPosition); \
UBO_FUNC(vec3, prevFrameCameraPosition); \
UBO_FUNC(int, prevWorldID);

#define PER_SAMPLE_UBO(UBO_FUNC) \
UBO_FUNC(vec3, cameraPosition); \
UBO_FUNC(vec3, baseFrameCameraPosition); \
UBO_FUNC(ivec2, cameraChunk); \
UBO_FUNC(ivec2, previousCameraChunk); \
UBO_FUNC(int, sampledFrameID); \
UBO_FUNC(float, distortionIntensity); \
UBO_FUNC(float, yaw); \
UBO_FUNC(float, roll); \
UBO_FUNC(float, pitch); \
UBO_FUNC(float, zoom); \
UBO_FUNC(int, flipY); \
UBO_FUNC(float, beatFromPos); \
UBO_FUNC(float, currentSpeed); \
UBO_FUNC(int, uWorldID); \
UBO_FUNC(ivec4, logicalFromPhysical); \
UBO_FUNC(vec3, sunDirection); \
UBO_FUNC(vec3, moonDirection); \
UBO_FUNC(vec3, sunIrradiance);

#if MAX_WORLD_COUNT > 4
#error "logicalFromPhysical is using ivec4 for mapping, this does not support more than 4 physical worlds"
#endif

#if !defined(CXX_STAGE)
layout(std140, binding = 1) uniform LAYOUTT_00 {
	PER_FRAME_CPU_UBO(UBO_DECLARE)
};


layout(std140, binding = 0) uniform LAYOUTT_0 {
	PER_SAMPLE_UBO(UBO_DECLARE)
};
struct PerSampleUniforms {
	PER_SAMPLE_UBO(UBO_DECLARE)
	vec4[6] padding;
};
layout(std140, binding = 14) buffer LAYOUTT_000 {
	PerSampleUniforms perSampleUbo[MAX_SAMPLE_COUNT];
};


const vec2 aspect = max(vec2(1.0), viewSize.xy / viewSize.yx);

layout(std430, binding = 2) buffer LAYOUTT_4 {
	uvec4 num_groups[3];
} computeIndirect;


layout(std430, binding = 3) buffer LAYOUTT_5 {
	uint bufferFront[MAX_WORLD_COUNT+1];
	uint bufferBack[MAX_WORLD_COUNT+1];
	uvec4 chunkID[];
};

layout(std430, binding = 4) buffer LAYOUTT_6 {
	BeatStructGPU[BEATS_COUNT] beatsSSBO;
};

layout(std430, binding = 6) buffer LAYOUTT__6 {
	int8_t voxelLightingSSBO[];
};

#define BEAT_(i) beatsSSBO[i].beat
#define BEAT_TYPE(i) beatsSSBO[i].type
#define PORTAL_TARGET(i) beatsSSBO[i].portalTarget

layout(std430, binding = 5) buffer LAYOUTT_8 {
	ivec4[] data;
} chunkIndirectCoordinates;

layout(std430, binding = 15) buffer LAYOUTT_7 {
	ivec2[] offset;
} atlasSSBO;

layout(std430, binding = 7) buffer LAYOUTT_888 {
	WorldRange[] worldRanges;
};

layout(std430, binding = 8) buffer LAYOUTT_8888 {
	LogicalID[] physicalFromLogical;
};

#define worldIdFailedToMap -1

int LogicalFromPhysical(int physicalID, int z) {
	for (int i = physicalID; i < LOGICAL_WORLD_COUNT; i += MAX_WORLD_COUNT) {
		WorldRange range = worldRanges[i];
		if (range.zStart == range.zEnd) break;
		if (range.zStart < z && z < range.zEnd) {
			return range.logicalWorldID;
		}
	}

	return worldIdFailedToMap;
}

int g_logicalWorldID = 0;
int g_physicalWorldID = 0;
int g_chunkImageJump = 0;

void SetPhysicalWorldID(int id) {
	g_physicalWorldID = id;
	g_chunkImageJump = id * WORLD_SIZE.y / 16;
}

bool SetLogicalWorldID(int physicalID, int z) {
	int logicalID = LogicalFromPhysical(physicalID, z);
	if (logicalID == worldIdFailedToMap) return false;
	
	SetPhysicalWorldID(physicalID);
	
	g_logicalWorldID = logicalID;

	return true;
}

void SetLogicalWorldID(int logicalID) {
	int physicalID = physicalFromLogical[logicalID].id;
	if (physicalID == worldIdFailedToMap) return;
	SetPhysicalWorldID(physicalID);
	g_logicalWorldID = logicalID;
}

void UpdateLogicalWorldID(uint blockID) {
	int logicalID = worldIdFailedToMap;
	if (blockID == id_portal_forward) {
		logicalID = physicalFromLogical[g_logicalWorldID].nextLogical;
	} else if (blockID == id_portal_backward) {
		logicalID = physicalFromLogical[g_logicalWorldID].prevLogical;
	}

	if (logicalID != worldIdFailedToMap) {
		SetLogicalWorldID(logicalID);
	}
}

int GetWaterHeight() {
	switch (g_logicalWorldID) {
	case WORLD_NAME(0): return WATER_HEIGHT;
	case WORLD_NAME(1): return WATER_HEIGHT;
	case WORLD_NAME(2): return WATER_HEIGHT;
	case WORLD_NAME(3): return WATER_HEIGHT;
	case WORLD_NAME(4): return WATER_HEIGHT;
	}
	return WATER_HEIGHT;
}

layout(binding = 1, rgba16i) uniform iimage3D chunkImage;
layout(binding = 0) uniform sampler2D noisetex;
layout(binding = 1) uniform sampler3D skyLUT;
layout(binding = 2) uniform sampler2D atlasTexture;
layout(binding = 3) uniform sampler2D lutTexture;
layout(binding = 0, r8ui) uniform uimage3D voxelImage;
layout(binding = 3, rgba32f) uniform image2D distortionReuseImage;
layout(binding = 9) uniform sampler2D Texture9;
layout(binding = 10) uniform sampler2D bloomTexture;
layout(binding = 12) uniform sampler2D Texture12;
layout(binding = 13) uniform sampler2D Texture13;
layout(binding = 14) uniform sampler2D accumTexture;
layout(binding = 15) uniform sampler2D frameTexture;
layout(binding = 11) uniform isampler1D atlasOffsetTexture;

#define ACCUM_POS ivec2(0, WORLD_SIZE.z)
#define PREV_POS ivec2(1, WORLD_SIZE.z)

#define pow2(x) ((x) * (x))
#define clamp01(x) (clamp(x, 0.0, 1.0))

float floor16(float x) { return x - mod(x, 16.0f); }
vec2 floor16(vec2 x) { return x - mod(x, vec2(16.0f)); }

vec3 hsv(vec3 c) {
	const vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 rgb(vec3 c) {
	const vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);

	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);

	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

ivec3 rirCoordPrev(ivec3 pos) {
	pos.xz = (pos.xz + previousCameraChunk) % WORLD_SIZE.xz;
	return pos;
}

ivec3 rirCoord(ivec3 pos) {
	pos.xz = (pos.xz + cameraChunk) % WORLD_SIZE.xz;
	return pos;
}

ivec3 rirCoord(ivec3 pos, ivec2 cameraChunk2) {
	pos.xz = (pos.xz + cameraChunk2) % WORLD_SIZE.xz;
	return pos;
}

ivec3 LodCoord(ivec3 pos, int lod) {
	pos.xyz = pos.xyz >> lod;

	if (lod >= 1) pos.y += int(SPARSE_Y);

	return pos;
}

ivec3 SparseChunkLoad(ivec3 pos) {
	pos.y += g_chunkImageJump;
	return imageLoad(chunkImage, pos).rgb;
}

void SparseChunkStore(ivec3 pos, ivec4 value) {
	pos.y += g_chunkImageJump;
	imageStore(chunkImage, pos, value);
}

bool OutOfBounds(ivec3 pos) {
	return any(greaterThanEqual(pos.xyz, WORLD_SIZE.xyz)) || any(lessThan(pos.xyz, ivec3(0)));
}

bool ChunkAllocated(ivec3 pos) {
	if (OutOfBounds(pos)) return false;
	pos = rirCoord(pos);

	if (SPARSE) { ivec3 pos2 = SparseChunkLoad(pos/16); if (pos2.x < 0) return false; }

	return true;
}

uint VoxelRead(ivec3 pos, int lod) {
	if (BOUNDS_CHECKING && OutOfBounds(pos)) { return 0; }
	pos = rirCoord(pos);

	if (SPARSE) { ivec3 pos2 = SparseChunkLoad(pos/16); if (pos2.x < 0) return 0; pos = pos2 * 16 + (pos % 16); }
	if (lod == 4) return 1;

	return imageLoad(voxelImage, LodCoord(pos, lod)).r;
}

const bool myfix = false;

uint ChunkDataRead(ivec3 pos) {
	if (myfix) return 0;
	if (OutOfBounds(pos)) { return 0; }
	pos = rirCoord(pos);

	pos.xyz = pos.xyz >> 4;
	pos.x += int(sparseChunkDims.x*16)/4;
	pos.y += int(SPARSE_Y);

	pos.y += g_physicalWorldID*WORLD_SIZE.y/16;

	return imageLoad(voxelImage, pos).r;
}


void VoxelWrite(ivec3 pos, uint data, int lod) {
	if (OutOfBounds(pos)) { return; }
	if (lod != 0 && lod != 2) return;
	pos = rirCoord(pos);

	if (SPARSE) { ivec3 pos2 = SparseChunkLoad(pos / 16); if (pos2.x < 0) return; pos = pos2 * 16 + (pos % 16); }

	imageStore(voxelImage, LodCoord(pos, lod), uvec4(data));
}

void ChunkDataWrite(ivec3 pos, uint data) {
	if (myfix) return;
	pos = rirCoord(pos);

	if (any(greaterThanEqual(pos.xyz, WORLD_SIZE.xyz))) return;
	if (any(lessThan(pos.xyz, ivec3(0)))) return;

	pos.xyz = pos.xyz >> 4;
	pos.x += int(sparseChunkDims.x*16)/4;
	pos.y += int(SPARSE_Y);

	pos.y += g_physicalWorldID*WORLD_SIZE.y/16;

	imageStore(voxelImage, pos, uvec4(data));
}

vec3 WorldToVoxelSpace(vec3 position) {
	vec3 WtoV = vec3(0.0);
	WtoV.y -= cameraPosition.y;
	WtoV.xz += (WORLD_SIZE.xz) + (-cameraPosition.xz - floor(-cameraPosition.xz / 16.0) * 16.0);
	return position + WtoV;
}

uint triple32(uint x) {
	// https://nullprogram.com/blog/2018/07/31/
	x ^= x >> 17;
	x *= 0xed5ad4bbu;
	x ^= x >> 11;
	x *= 0xac4c1b51u;
	x ^= x >> 15;
	x *= 0x31848babu;
	x ^= x >> 14;
	return x;
}

float WangHash(uint seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return float(seed) / 4294967296.0;
}

vec2 WangHash(uvec2 seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return vec2(seed) / 4294967296.0;
}

float RandF(uint  seed) { return float(triple32(seed)) / float(0xffffffffu); }
vec2  Rand2F(uvec2 seed) { return vec2(triple32(seed.x), triple32(seed.y)) / float(0xffffffffu); }

uint randState = triple32(0);
uint RandNext() { return randState = triple32(randState); }
uvec2 RandNext2() { return uvec2(RandNext(), RandNext()); }
uvec3 RandNext3() { return uvec3(RandNext2(), RandNext()); }
uvec4 RandNext4() { return uvec4(RandNext3(), RandNext()); }
float RandNextF() { return float(RandNext()) / float(0xffffffffu); }
vec2 RandNext2F() { return vec2(RandNext2()) / float(0xffffffffu); }
vec3 RandNext3F() { return vec3(RandNext3()) / float(0xffffffffu); }
vec4 RandNext4F() { return vec4(RandNext4()) / float(0xffffffffu); }

float RandNextF(uint seed) { return float(triple32(seed)) / float(0xffffffffu); }

vec2 TAAHash(uint seed) {
	//return vec2(0.0);
	return (Rand2F(floatBitsToUint(seed) * uvec2(12345, 12345 * 2) + uvec2(0, 1)) - 0.5) / viewSize * 2.0;
}

vec2 cx_mul(vec2 a, vec2 b) {
	return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 cx_div(vec2 a, vec2 b) {
	return vec2(((a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y)), ((a.y * b.x - a.x * b.y) / (b.x * b.x + b.y * b.y)));
}

vec2 cx_sin(vec2 a) {
	return vec2(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y));
}

vec2 cx_cos(vec2 a) {
	return vec2(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y));
}

vec2 cx_tan(vec2 a) { return cx_div(cx_sin(a), cx_cos(a)); }
vec2 cx_log(vec2 a) {
	float rpart = sqrt((a.x * a.x) + (a.y * a.y));
	float ipart = atan(a.y, a.x);
	if (ipart > PI) ipart = ipart - (2.0 * PI);
	return vec2(log(rpart), ipart);
}

vec2 cx_exp(vec2 a) {
	return cx_mul(vec2(exp(a.x), 0.0), vec2(cos(a.y), sin(a.y)));
}

vec2 cx_pow(vec2 a, vec2 b) {
	return cx_exp(cx_log(a) + b);
}

vec2 as_polar(vec2 z) {
	return vec2(
		length(z),
		atan(z.y, z.x)
	);
}
vec2 cx_pow(vec2 v, float p) {
	vec2 z = as_polar(v);
	return pow(z.x, p) * vec2(cos(z.y * p), sin(z.y * p));
}

float im(vec2 z) {
	return ((atan(z.y, z.x) / PI) + 1.0) * 0.5;
}

float cx_abs(vec2 a) {
	// vector length
	return sqrt(a.x * a.x + a.y * a.y);
}

vec2 cx_norm(vec2 a) {
	return a / vec2(cx_abs(a));
}

vec2 cx_sqrt(vec2 z) {
	return vec2(sqrt((cx_abs(z) + z.x) / 2.0f), z.y * sqrt((cx_abs(z) - z.x) / 2.0f) * sign(z.y));
}

vec3 random3(vec3 c) {
	float j = 4096.0 * sin(dot(c, vec3(17.0, 59.4, 15.0)));
	vec3 r;
	r.z = fract(512.0 * j);
	j *= .125;
	r.x = fract(512.0 * j);
	j *= .125;
	r.y = fract(512.0 * j);
	return r - 0.5;
}

float NormalizeCDF(float x) {
	// Approximate CDF of standard normal distribution (mean = 0, stddev = 1)
	return tanh(0.79788456 * (x + 0.044715 * pow(x, 3.0)));
}

/* skew constants for 3d simplex functions */
const float F3 = 0.3333333;
const float G3 = 0.1666667;

/* 3d simplex noise */
float simplex3d(vec3 p) {
	//if ((int(abs(p.z)) % 1024) < 512) p.z = 12456;
	//p.z = 12345;

	/* 1. find current tetrahedron T and it's four vertices */
	/* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
	/* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/

	/* calculate s and x */
	vec3 s = floor(p + dot(p, vec3(F3)));
	vec3 x = p - s + dot(s, vec3(G3));

	/* calculate i1 and i2 */
	vec3 e = step(vec3(0.0), x - x.yzx);
	vec3 i1 = e * (1.0 - e.zxy);
	vec3 i2 = 1.0 - e.zxy * (1.0 - e);

	/* x1, x2, x3 */
	vec3 x1 = x - i1 + G3;
	vec3 x2 = x - i2 + 2.0 * G3;
	vec3 x3 = x - 1.0 + 3.0 * G3;

	/* 2. find four surflets and store them in d */
	vec4 w, d;

	/* calculate surflet weights */
	w.x = dot(x, x);
	w.y = dot(x1, x1);
	w.z = dot(x2, x2);
	w.w = dot(x3, x3);

	/* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
	w = max(0.6 - w, 0.0);

	/* calculate surflet components */
	d.x = dot(random3(s), x);
	d.y = dot(random3(s + i1), x1);
	d.z = dot(random3(s + i2), x2);
	d.w = dot(random3(s + 1.0), x3);

	/* multiply d by w^4 */
	w *= w;
	w *= w;
	d *= w;

	/* 3. return the sum of the four surflets */
	return NormalizeCDF(dot(d, vec4(52.0))*4.0);
}

mat2 rotate(float rad) {
	return mat2(cos(-rad), -sin(-rad), sin(-rad), cos(-rad));
}

/* const matrices for 3d rotation */
const mat3 rot1 = mat3(-0.37, 0.36, 0.85, -0.14, -0.93, 0.34, 0.92, 0.01, 0.4);
const mat3 rot2 = mat3(-0.55, -0.39, 0.74, 0.33, -0.91, -0.24, 0.77, 0.12, 0.63);
const mat3 rot3 = mat3(-0.71, 0.52, -0.47, -0.08, -0.72, -0.68, -0.7, -0.45, 0.56);

/* directional artifacts can be reduced by rotating each octave */
float simplex3d_fractal(vec3 m) {
	return (0.5333333 * simplex3d(m * rot1)
		+ 0.2666667 * simplex3d(2.0 * m * rot2)
		+ 0.1333333 * simplex3d(4.0 * m * rot3)
		+ 0.0666667 * simplex3d(8.0 * m));
}

float NewSimplex(vec3 m) {
	return (0.5333333 * simplex3d(m)
		+ 0.2666667 * simplex3d(2.0 * m)
		+ 0.1333333 * simplex3d(4.0 * m)
		+ 0.0666667 * simplex3d(8.0 * m))*0.5+0.5;
}

vec3 VoxelToWorld(vec3 pos) {
	vec2 chunk = cameraPosition.xz - mod(cameraPosition.xz, 16.0);

	pos.xyz -= offsetInStructure;
	pos.xz += chunk;
	pos.y += WORLD_SIZE.y / 2;

	return pos;
}

vec3 WorldToVoxel(vec3 pos) {
	pos.y -= WORLD_SIZE.y / 2;
	pos.xz -= cameraPosition.xz - mod(cameraPosition.xz, 16.0);
	pos.xyz += offsetInStructure;

	return pos;
}

vec3 WorldToVoxel(vec3 pos, vec3 camPos) {
	pos.y -= WORLD_SIZE.y / 2;
	pos.xz -= camPos.xz - mod(camPos.xz, 16.0);
	pos.xyz += offsetInStructure;

	return pos;
}

vec3 PrevVoxelToWorld(vec3 pos) {
	vec2 chunk = prevRegenCameraPosition.xz - mod(prevRegenCameraPosition.xz, 16.0);

	pos.xyz -= offsetInStructure;
	pos.xz += chunk;
	pos.y += WORLD_SIZE.y / 2;

	return pos;
}

vec3 BlockColor(uint data, vec3 voxelPos) {
	float height = simplex3d_fractal(voxelPos / vec3(2048)) * 0.5 + 0.5;
	height = 0.05 + height * 0.33;

	if (data == id_sand) return vec3(0.8);
	if (data == id_leaves) {
		

		return rgb(vec3(height, LEAVES_HSV.gb));

		//return rgb(LEAVES_HSV * vec3(1.0, 1.0 - 0 * interp(voxelPos.y, 128.0, 200.0) * 0.8, 1.0));
	}

	if (data == id_grass_top) {
		return rgb(vec3(height, GRASS_HSV.gb));
		return rgb(GRASS_HSV);
	}

	return vec3(1.0);
}

vec4 BlockTexture(vec2 uv, uint blockID, uint faceIndex, vec3 voxelPos) {
	voxelPos = VoxelToWorld(voxelPos);

	if (blockID == id_grass && faceIndex == 2) {
		blockID = id_dirt;
	}

	if (blockID == id_grass && faceIndex == 5) {
		blockID = id_grass_top;
	}

	if (blockID == id_oak_log && (faceIndex == 2 || faceIndex == 5)) {
		blockID = id_oak_log_top;
	}

	if (IsTorch(blockID)) {
		blockID = id_torch;
	}

	if (blockID == id_stone2) blockID = id_stone;

	//vec2 offset = atlasSSBO.offset[blockID];
	vec2 offset = texelFetch(atlasOffsetTexture, int(blockID), 0).rg;
	vec4 color = textureLod(atlasTexture, (uv * 16.0 + offset) / 1024.0, 0);

	if ((blockID == id_grass && color.a == 1.0) || blockID == id_grass_top) {
		blockID = id_grass_top;
	}

	color.rgb *= clamp(BlockColor(blockID, voxelPos), 0.0, 1.0);
	color.rgb = pow(color.rgb, vec3(TEXTURE_GAMMA));

	return color;
}

#define Safepow(x, y) (pow(abs(x),y) * sign(x))
float cubesmooth(float x) {
	return x * x * (3.0 - 2.0 * x);
}

float cubesmooth01(float x) {
	x = clamp(x, 0.0, 1.0);
	return x * x * (3.0 - 2.0 * x);
}

vec2 cubesmooth(vec2 x) {
	return x * x * (3.0 - 2.0 * x);
}

float EaseInSin(float x) {
	return 1.0 - cos((x * 3.14159) / 2.0);
}

float EaseOutSin(float x) {
	return sin((x * 3.14159) / 2.0);
}

float EaseInOutSin(float x) {
	return -(cos(x * 3.14159) - 1.0) / 2.0;
}

// Return the index of the nearest beat less than or equal to the target
int BinarySearchGT(int target) {
	int high = BEATS_COUNT - 1;

	int low = 0;

	while (low <= high) {
		int mid = (high + low) / 2;

		int value = int(beatsSSBO[mid].zPos);

		if (value == target) return mid;
		if (value > target) high = mid - 1; // if (value < target) low = mid + 1;
		else low = mid + 1;                 // else high = mid - 1;
	}

	if (target >= int(beatsSSBO[high].zPos)) { // <=
		return high; // return low;
	}

	return -1;
}

int BinarySearchNearest(int target) {
	int i = BinarySearchGT(target);
	int x1 = int(beatsSSBO[i].zPos);

	int x2 = int(beatsSSBO[i+1].zPos);
	return abs(x1 - target) < abs(x2 - target) ? i : i + 1;
}

bool BinarySearchIsExact(int target, int i) {
	int x1 = int(beatsSSBO[i].zPos);
	return target == x1;
}

//#include "Animations.glsl"

#endif
