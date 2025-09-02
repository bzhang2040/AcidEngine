#ifdef CXX_STAGE
    #define InitBeats_glsl "Uniforms.glsl", "INIT_BEATS_STAGE", "compute"
#endif

#ifdef INIT_BEATS_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= BEATS_COUNT) { return; }

    beatsSSBO[tid].zPos = GetCameraPos(GetTimeFromBeat(beatsSSBO[tid].beat)).z;
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

float DistortionIntensity() {
    //return 1.0;
    if (!DO_DISTORTION) return 0.0;
    float prev = 0.0;
    float curr = 0.0;
    float temp = 0.0;
    float beat = GetBeatFromTime(timeFromPos);

    Key(0.2, 265, 271, powf(0.75, temp));
    Key(0.6, 271, 275, powf(0.6, cubesmooth(temp)));
    Key(0.8, 277, 313, temp);
    Key(0.0, 500, 505, powf(4.0, temp));

    return curr;
}

uint VoxelRead2(ivec3 pos, ivec2 cameraChunk2) {
    pos = rirCoord(pos, cameraChunk2);

    if (SPARSE) { ivec3 pos2 = SparseChunkLoad(pos / 16); if (pos2.x < 0) return 0; pos = pos2 * 16 + (pos % 16); }

    return imageLoad(voxelImage, LodCoord(pos, 0)).r;
}

float GetTime(int tid) {
    return TIME_OFFSET + nonBlurTime + (tid * SHUTTER_ANGLE(nonBlurTime)) / framerate / sampleCount * float(!interactive);
}

vec3 GetCameraPosition(int tid) {
    return GetCameraPos(GetTime(tid)) - currMovement;
}

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= sampleCount) { return; }

    PerSampleUniforms u;

    u.time = GetTime(tid);

    u.cameraPosition = GetCameraPosition(tid);

    u.cameraChunk = ivec2(floor16(u.cameraPosition.xz)) + ivec2(WORLD_SIZE.x, WORLD_SIZE.z) * 1024;
    u.previousCameraChunk = ivec2(-floor16(prevRegenCameraPosition.xz)) + ivec2(WORLD_SIZE.x, WORLD_SIZE.z) * 1024;

    ivec2 ebin = ivec2(floor16(prevFrameCameraPosition.xz)) + ivec2(WORLD_SIZE.x, WORLD_SIZE.z) * 1024;

    
    if (resetCamera == 1) {
        for (int i = 0; i < PORTAL_COUNT; ++i) {
            if (GetCameraPos(nonBlurTime+TIME_OFFSET).z < worldRanges[i].zEnd) {
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

    u.timeFromPos = GetTimeFromPos(u.baseFrameCameraPosition.z);

    u.currentSpeed = GetCameraPos(u.timeFromPos + 1.0).z - GetCameraPos(u.timeFromPos).z;

    u.sunDirection = SunDirection(u.timeFromPos);
    u.moonDirection = MoonDirection(u.timeFromPos);
    u.sunIrradiance = GetSunIrradiance(kPoint(vec3(0.0) + u.cameraPosition), u.sunDirection);

    perSampleUbo[tid] = u;
};

#endif
#endif