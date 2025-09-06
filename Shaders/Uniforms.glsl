#ifdef CXX_STAGE
    #define InitBeats_glsl "Uniforms.glsl", "INIT_BEATS_STAGE", "compute"
#endif

#ifdef INIT_BEATS_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid < BEATS_COUNT) {
        beatsSSBO[tid].zPos = GetCameraPos(beatsSSBO[tid].beat).z;
    }

    if (tid < PORTAL_COUNT) {
        //beatsSSBO[tid].zPos = GetCameraPos(beatsSSBO[tid].beat).z;
        //worldRanges[i].zEnd

        if (tid == 0) {
            worldRanges[tid].zStart = -10000000;
            worldRanges[tid].zEnd = int(GetBeatPos(worldRanges[tid+1].beat));
        } else if (tid == PORTAL_COUNT-1) {
            worldRanges[tid].zStart = int(GetBeatPos(worldRanges[tid].beat));
            worldRanges[tid].zEnd = 100000000;
        } else {
            worldRanges[tid].zStart = int(GetBeatPos(worldRanges[tid].beat));
            worldRanges[tid].zEnd = int(GetBeatPos(worldRanges[tid+1].beat));
        }

        worldRanges[tid].zStart -= WORLD_SIZE.z / 2 + 32;
        worldRanges[tid].zEnd += WORLD_SIZE.z / 2 + 32;
    }
};

#endif
#endif






#ifdef CXX_STAGE
    #define Uniforms_glsl "Uniforms.glsl", "GENERATE_UNIFORMS_STAGE", "compute"
#endif

#ifdef GENERATE_UNIFORMS_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

//#include SKY

uint VoxelRead2(ivec3 pos, ivec2 cameraChunk2) {
    pos = rirCoord(pos, cameraChunk2);

    if (SPARSE) { ivec3 pos2 = SparseChunkLoad(pos / 16); if (pos2.x < 0) return 0; pos = pos2 * 16 + (pos % 16); }

    return imageLoad(voxelImage, LodCoord(pos, 0)).r;
}

vec3 GetCameraPosition(int tid) {
    float time = nonBlurTime + (tid * SHUTTER_ANGLE(nonBlurBeat)) / framerate / sampleCount * float(!interactive);

    return GetCameraPos(GetBeatFromTime(time)) - currMovement;
}

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= sampleCount) { return; }

    PerSampleUniforms u;

    u.cameraPosition = GetCameraPosition(tid);

    u.cameraChunk = ivec2(floor16(u.cameraPosition.xz)) + ivec2(WORLD_SIZE.x, WORLD_SIZE.z) * 1024;
    u.previousCameraChunk = ivec2(-floor16(prevRegenCameraPosition.xz)) + ivec2(WORLD_SIZE.x, WORLD_SIZE.z) * 1024;

    ivec2 ebin = ivec2(floor16(prevFrameCameraPosition.xz)) + ivec2(WORLD_SIZE.x, WORLD_SIZE.z) * 1024;

    
    if (resetCamera == 1) {
        for (int i = 0; i < PORTAL_COUNT; ++i) {
            if (GetCameraPos(nonBlurBeat).z < worldRanges[i].zEnd) {
                u.uWorldID = worldRanges[i].logicalWorldID;
                break;
            }
        }
    } else {
        u.uWorldID = prevWorldID;
    }

    SetLogicalWorldID(u.uWorldID);

    ivec3 voxelEnd = ivec3(WorldToVoxel(u.cameraPosition, prevFrameCameraPosition));
    ivec3 voxelStart = ivec3(WorldToVoxel(prevFrameCameraPosition, prevFrameCameraPosition));
    int sng = int(sign(voxelEnd.z-voxelStart.z));
    int count = min(100, abs(voxelEnd.z - voxelStart.z));
    
    for (int i = 1; i <= count; ++i) {
        if (IsPortal(VoxelRead2(voxelStart + ivec3(0,0,1)*i*int(sign(voxelEnd.z-voxelStart.z)), ebin))
            && !IsPortal(VoxelRead2(voxelStart + ivec3(0,0,1)*(i-1)*int(sign(voxelEnd.z-voxelStart.z)), ebin))
        ) {
            UpdateLogicalWorldID(VoxelRead2(voxelStart + ivec3(0,0,1)*i*int(sign(voxelEnd.z-voxelStart.z)), ebin));
            u.uWorldID = g_logicalWorldID;
            break;
        }
    }

    u.baseFrameCameraPosition = GetCameraPosition(0);

    u.sampledFrameID = frameID * sampleCount + tid;

    u.distortionIntensity = DistortionIntensity();

    u.beatFromPos = GetBeatFromPos(u.baseFrameCameraPosition.z);

    u.currentSpeed = GetCameraPos(u.beatFromPos + GetBeatFromTime(1.0)).z - GetCameraPos(u.beatFromPos).z;

    u.sunDirection = SunDirection(u.beatFromPos);
    u.moonDirection = MoonDirection(u.beatFromPos);
    u.sunIrradiance = GetSunIrradiance(kPoint(vec3(0.0) + u.cameraPosition), u.sunDirection);

    perSampleUbo[tid] = u;
};

#endif
#endif