
const vec2 trackPos = vec2(0.5, 92.5);
const float customyaw = 0.0000001;
const float custompitch = -0.001;
const float customzoom = 0.0;
const vec3 custommovement = vec3(0.0, 0.0, 0.0) * vec3(1.0) + vec3(0.0001);

#define UPDATE_INDIRECT true

#define SPARSE true
#define SPARSITY 4

#define RASTER false

#define BOUNDS_CHECKING false

const ivec3 WORLD_BITS = ivec3(10, 9, 10);
const ivec3 WORLD_SIZE = ivec3(1) << WORLD_BITS;

#define SUB_VOXEL_TRACE true
#define TRACE_LEAVES false

#define MAX_WORLD_COUNT 4
#define denseChunkDims ivec3(WORLD_SIZE.x/16, WORLD_SIZE.y/16, WORLD_SIZE.z/16)
#define sparseChunkDims ((denseChunkDims * ivec3(1,MAX_WORLD_COUNT,1)) / ivec3(1,1,SPARSITY))
const int sparseTotalSize = sparseChunkDims.x * sparseChunkDims.y * sparseChunkDims.z;

#define SPARSE_Y (sparseChunkDims.y*16)

#define MAX_LOGICAL_WORLD_COUNT 16

#define windowWidth (1280)
#define windowHeight (720)

#define viewSize vec2(windowWidth, windowHeight)

#define DO_FISHEYE false
#define DO_CLOUDS true
#define DO_FOG true
#define DO_ATMOSPHERE false
#define FOG_START 0.2
#define DO_DISTORTION true
#define FAST_LIGHTING false

#define CURVATURE_SAMPLES 1

#define writeFrames true
#define interactive (true && (!writeFrames))
float maxSeconds = 90.0;    // Full video = 485.0
#define SAMPLE_COUNT (writeFrames ? 16 : 1)
#define encodeVideo (true && writeFrames)
#define MAX_SAMPLE_COUNT 512
#define TRILINEAR_TERRAIN (false || writeFrames)

#define START_FRAME (1 + 3282 + 3741*0 + 5820*0 + 6955*0 + 9125*0 + 10745*0 + 14000*0 + 14900*0 + 16000*0 + 17400*0 + 24150*0)

#define EXPOSURE  1.0
#define EXPOSURE2 2.0
#define GAMMA 2.2

#define TEXTURE_GAMMA 2.2
#define LEAVES_HSV vec3(0.30, 0.85, 0.9)
#define GRASS_HSV vec3(0.25, 0.60, 0.85)
#define TREE_PROBABILITY 0.999

#define SKY_MULT 6.0

#define WATER_HEIGHT 80
#define SAND_HEIGHT (WATER_HEIGHT + 3)

#define LOCAL_LOD 0
#define LOD_STEP 2
#define MAX_LOD 4

const float framerate = 60.0;
#define FOV 90.0f

const vec3 offsetInStructure = vec3(ivec3(vec3(WORLD_SIZE) * vec3(0.5, 0.5, 0.5) + vec3(0.0, 0.0, 0.0)));

#define VIDEO_LENGTH_SECONDS (9*60)

#define BLOCKS_PER_SECOND 80.0f
#define BEATS_PER_MINUTE 160.0f
#define BEATS_PER_SECOND (BEATS_PER_MINUTE / 60.0f)

float GetBeatFromTime(float time) {
    float secondsPerBeat = 60.0 / BEATS_PER_MINUTE;

    return time / secondsPerBeat;
}

float GetTimeFromBeat(float beat) {
    float secondsPerBeat = 60.0 / BEATS_PER_MINUTE;

    return beat * secondsPerBeat;
}

//#include INCLUDE
