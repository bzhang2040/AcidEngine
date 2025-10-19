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

    if (tid < LOGICAL_WORLD_COUNT) {
        //beatsSSBO[tid].zPos = GetCameraPos(beatsSSBO[tid].beat).z;
        //worldRanges[i].zEnd

        if (tid == 0) {
            worldRanges[tid].zStart = -10000000;
            worldRanges[tid].zEnd = int(GetBeatPos(worldRanges[tid+1].beat));
        } else if (tid == LOGICAL_WORLD_COUNT -1) {
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

vec3 SunDirection(float beat) {
    float sunAngle = GetSunAngle(beat);
    float sunRotation = 30.0;

    float curr = 45.0;

    vec3 sunDir = vec3(0.0, 0.0, 1.0);

    vec2 v = vec2(sunDir.x, sunDir.z) * rotate(radians(sunRotation));
    sunDir.x = v.x;
    sunDir.z = v.y;
    v = vec2(sunDir.y, sunDir.z) * rotate(radians(sunAngle));
    sunDir.y = v.x;
    sunDir.z = v.y;

    return normalize(sunDir);
}

vec3 MoonDirection(float beat) {
    vec3 moonDirection = SunDirection(beat);
    vec2 v = vec2(moonDirection.x, moonDirection.z) * rotate(radians(-30.0));
    moonDirection.x = v.x;
    moonDirection.z = v.y;
    moonDirection.y *= -1.0;
    moonDirection = -SunDirection(beat);

    return moonDirection;
}

uint VoxelRead2(ivec3 pos, ivec2 cameraChunk2) {
    pos = rirCoord(pos, cameraChunk2);

    if (SPARSE) { ivec3 pos2 = SparseChunkLoad(pos / 16); if (pos2.x < 0) return 0; pos = pos2 * 16 + (pos % 16); }

    return imageLoad(voxelImage, LodCoord(pos, 0)).r;
}

vec3 GetCameraPosition(int tid) {
    float time = nonBlurTime + (tid * SHUTTER_ANGLE(nonBlurBeat)) / framerate / sampleCount * float(!interactive);

    return GetCameraPos(GetBeatFromTime(time)) - currMovement;
}

float GetBeatFromPos(float pos) { // Inverts the function using newtons method.
    float t = 0.0;

    for (int i = 0; i < 100; ++i) {
        float f = GetCameraPos(t).z - pos;
        float df = GetCameraPos(t + GetBeatFromTime(1.0)).z - GetCameraPos(t).z;

        if (abs(f) < 0.0001 || df == 0.0) {
            return t;
        }

        t -= f / df;
    }

    return t;
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
        for (int i = 0; i < LOGICAL_WORLD_COUNT; ++i) {
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

    u.cameraPosition.y -= GetWaterHeight() + 1;
    u.roll = (1 - GetLatent2()) * radians(-180);

    u.flipY = 0;
    if (u.cameraPosition.y < 0.0) {
        u.cameraPosition.y = -u.cameraPosition.y;
        u.flipY = 1;
    }

    u.cameraPosition.y += GetWaterHeight() + 1;

    u.baseFrameCameraPosition = GetCameraPosition(0);

    u.sampledFrameID = frameID * sampleCount + tid;

    u.beatFromPos = GetBeatFromPos(u.baseFrameCameraPosition.z);

    u.distortionIntensity = DistortionIntensity(u.beatFromPos);
    
    u.currentSpeed = GetCameraPos(u.beatFromPos + GetBeatFromTime(1.0)).z - GetCameraPos(u.beatFromPos).z;

    u.yaw = cpu_yaw + GetYaw();
    u.pitch = cpu_pitch + GetPitch();
    u.zoom = cpu_zoom;

    u.sunDirection = SunDirection(u.beatFromPos);
    u.moonDirection = MoonDirection(u.beatFromPos);
    u.sunIrradiance = GetSunIrradiance(kPoint(vec3(0.0) + u.cameraPosition), u.sunDirection);

    for (int physicalID = 0; physicalID < MAX_WORLD_COUNT; ++physicalID) {
        int minZCoord = int(u.baseFrameCameraPosition.z) - WORLD_SIZE.z / 2 + 32;
        int maxZCoord = int(u.baseFrameCameraPosition.z) + WORLD_SIZE.z / 2 + 32;

        int logicalWorldID = LogicalFromPhysical(physicalID, minZCoord);

        if (logicalWorldID == worldIdFailedToMap) {
            logicalWorldID = LogicalFromPhysical(physicalID, maxZCoord);
        }

        u.logicalFromPhysical[physicalID] = logicalWorldID;
    }

    perSampleUbo[tid] = u;
};

#endif
#endif