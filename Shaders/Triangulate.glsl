
#if !defined CXX_STAGE

#define IDWORLD 0
#define DIWORLD(id) (id % sparseTotalSize)

#define trackDist (abs(position.xy - (trackPos.xy + vec2(0, 1)) + 0.5))
#define trackDelta (position.xy - (trackPos.xy + vec2(0, 1)) + 0.5)

// 1x2 tunnel. The tiny 1-block wide air tunnel
#define SmallestAirTunnel() bool(trackDist.x < 1 + 0 && (abs(position.y - trackPos.y - 1.0)) < 1.5)

// 3x3 tunnel. 1 block on either side and 1 block above.
#define MediumAirTunnel() bool(trackDist.x < 2 + 0 && (abs(position.y - trackPos.y - 1.5)) < 2.0)

#define MediumAirTunnel2() bool(trackDist.x < 2 + 0 && (abs(position.y - trackPos.y - 1.5)) < 2.0)
#define MediumAirTunnelBorder2() (!MediumAirTunnel2() && bool(trackDist.x < 3 + 0 && (abs(position.y - trackPos.y - 1.5)) < 3.0))

const float beatRadius = 2.0;

const vec2 cPos = vec2(trackPos.x, trackPos.y) + vec2(-0.5, 2.0);
#define freq ((BLOCKS_PER_SECOND * (60.0 / BEATS_PER_MINUTE)) / 2.0)
#define crunch(x, y) (floor((x) / vec3(y)) * vec3(y))

bool torchSection(int idx) {
    //return true;
    return BEAT_(idx) < 313 || BEAT_(idx) >= 361.0;
}

#endif


#ifdef CXX_STAGE
    #define TriangleInit_glsl "Triangulate.glsl", "TRIANGLE_INIT_STAGE", "compute"
#endif

#ifdef TRIANGLE_INIT_STAGE
#ifdef COMPUTE_STAGE

layout (local_size_x = 256) in;

void main() {
    int idx = int(gl_GlobalInvocationID.x);

    if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.x == 0) {
        computeIndirect.num_groups[chunkUpdates].x = 0;
        computeIndirect.num_groups[chunkUpdates].y = MAX_WORLD_COUNT;
        computeIndirect.num_groups[chunkUpdates].z = 1;
    }
};

#endif
#endif



#ifdef CXX_STAGE
    #define InitChunks0_glsl "Triangulate.glsl", "INIT_CHUNKS0_STAGE", "compute"
#endif

#ifdef INIT_CHUNKS0_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

void main() {
    for (int i = 0; i < MAX_WORLD_COUNT; ++i) {
        SetPhysicalWorldID(i); SparseChunkStore(ivec3(gl_GlobalInvocationID), ivec4(-1));

        bufferFront[IDWORLD] = 0;
        bufferBack[IDWORLD] = 0;
    }
}
#endif
#endif

#ifdef CXX_STAGE
    #define InitChunks_glsl "Triangulate.glsl", "INIT_CHUNKS_STAGE", "compute"
#endif

#ifdef INIT_CHUNKS_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

uint WarpAtomicAdd() {
    uint liveMask = uint(ballotARB(true));
    uint liveCount = bitCount(liveMask);

    uint prefixSum = bitCount(liveMask & ((1 << gl_SubGroupInvocationARB) - 1));

    uint first_thread = findLSB(liveMask);

    uint vertID = 0;

    if (gl_SubGroupInvocationARB == first_thread) {
        vertID = atomicAdd(bufferBack[IDWORLD], liveCount);
    }

    return readInvocationARB(vertID, first_thread) + int(prefixSum);
}

int Linearizer(ivec3 pos) {
    //return int(WarpAtomicAdd());

    ivec3 worldSize = sparseChunkDims;
    //return (pos.z + pos.y * worldSize.z + pos.x * worldSize.z * worldSize.y);
    //return (pos.x + pos.y * worldSize.x + pos.z * worldSize.x * worldSize.y);
    int id = 0;
    id += (pos.x & 1) << 0;
    id += (pos.y & 1) << 1;
    id += (pos.z & 1) << 2;
    pos.xyz = pos.xyz >> 1;
    worldSize.xyz /= 2;
    //id += (pos.x + pos.y * worldSize.x + pos.z * worldSize.x * worldSize.y) << 3;
    id += (pos.x & 1) << 3;
    id += (pos.y & 1) << 4;
    id += (pos.z & 1) << 5;
    pos.xyz = pos.xyz >> 1;
    worldSize.xyz /= 2;
    //id += (pos.x + pos.y * worldSize.x + pos.z * worldSize.x * worldSize.y) << 6;
    id += (pos.x & 1) << 6;
    id += (pos.y & 1) << 7;
    id += (pos.z & 1) << 8;
    pos.xyz = pos.xyz >> 1;
    worldSize.xyz /= 2;
    id += (pos.x & 1) << 9;
    id += (pos.y & 1) << 10;
    id += (pos.z & 1) << 11;
    pos.xyz = pos.xyz >> 1;
    worldSize.xyz /= 2;
    id += (pos.x + pos.y * worldSize.x + pos.z * worldSize.x * worldSize.y) << 12;
    return id;
}

// Zero out all of the sparse chunks
void main1() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int id = Linearizer(pos);

    //if (id >= sparseTotalSize) return;
    chunkID[DIWORLD(id)].xyz = uvec3(pos);
};

void main() {
    for (int i = 0; i < MAX_WORLD_COUNT; ++i) {
        SetPhysicalWorldID(i); main1();
    }
}
#endif
#endif


#ifndef CXX_STAGE
bool ChunkChanged(ivec3 tid) {
    return bool(shaderReload) || floor(VoxelToWorld(tid)) != floor(PrevVoxelToWorld(rirCoordPrev(rirCoord(tid))));
}
#endif


#ifdef CXX_STAGE
#define ComputeChunkUpdates_glsl "Triangulate.glsl", "COMPUTE_CHUNK_UPDATES", "compute"
#endif

#ifdef COMPUTE_CHUNK_UPDATES
#ifdef COMPUTE_STAGE

layout(local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

void main() {
    ivec3 tid = ivec3(gl_GlobalInvocationID);

    if (!ChunkChanged(tid * 16)) {
        return;
    }

    uint bumpIndex = atomicAdd(computeIndirect.num_groups[chunkUpdates].x, 1u);
    chunkIndirectCoordinates.data[bumpIndex].xyz = tid;
};
#endif
#endif



#ifdef CXX_STAGE
#define ComputeDense_glsl "Triangulate.glsl", "COMPUTE_DENSE_STAGE", "compute"
#endif

#ifdef COMPUTE_DENSE_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

//#include "Beats.glsl"

int CheckPosition(vec3 position, int idx, bool exact) {
    ivec3 pos = ivec3(position) - ivec3(trackPos.xy, 0);
    
    int allBeats = AllBeats(pos, false);
    if (allBeats != 0) {
        return allBeats;
    }
    
    int beatType = BEAT_TYPE(idx);
    
    if (beatType == beat_type_portal) return 0;

    if (beatType == beat_type_nothing) return 0;
    
    if (beatType == beat_type_programmatic) {
        int height = (2 + idx) % 4;
        if (exact && CheckExtent(pos, 2, 2, height+1, height+1, id_both)) return id_torch;
        
        if (CheckExtent(pos, 2, 2, height, height, id_both)) return id_permastone;
        return 0;
    }

    if (!exact) return 0;

    int beatSide = beatType >= 20 ? id_both : (beatType >= 10 ? id_left : (id_right));
    beatType %= 10;
    bool blockRightSide = position.x - trackPos.x > 0.0;
    

    if (beatType == r_low) {
        if (CheckExtent(pos, 1, 1, 0, 0, beatSide)) return id_permastone;
        if (CheckExtent(pos, 1, 1, 1, 1, beatSide)) return id_torch;
    } else if (beatType == r_default) {
        if (CheckExtent(pos, 1, 1, 0, 1, beatSide)) return id_permastone;
        if (CheckExtent(pos, 1, 1, 2, 2, beatSide)) return id_torch;
    } else if (beatType == r_wide) {
        if (CheckExtent(pos, 2, 2, 2, 2, beatSide)) return id_permastone;
        if (CheckExtent(pos, 1, 1, 2, 2, beatSide)) return blockRightSide ? id_torch_right : id_torch_left;
    } else {
        if (CheckExtent(pos, 1, 1, 0, 1, id_both)) return id_permastone;
        if (CheckExtent(pos, 1, 1, 2, 2, id_both)) return id_torch;
    }

    return 0;
}

//#include "Terrain.glsl"

shared bool filled = false;
shared bool fullyOpaque = true;
shared uint8_t[18][18][18] sharedData;

vec3 GetNormal2(int faceIndex) {
    if (faceIndex == 0) { return vec3(0, 0, -1); }
    if (faceIndex == 1) { return vec3(-1, 0, 0); }
    if (faceIndex == 2) { return vec3(0, -1, 0); }
    if (faceIndex == 3) { return vec3(0, 0, 1); }
    if (faceIndex == 4) { return vec3(1, 0, 0); }
    if (faceIndex == 5) { return vec3(0, 1, 0); }
    return vec3(0.0);
}

bool VoxelIsAir(uint data) {
    return data == 0 || IsPortal(data);
}

void main1(ivec3 tid) {
    filled = false;
    fullyOpaque = true;
    memoryBarrierShared();
    barrier();

    {
        ivec3 global = (tid / 16) * 16;
        ivec3 local = (tid % 16);
        uint data = 0;

        // Y
        data = VoxelIsFilled(VoxelToWorld(global + local.xyz + ivec3(0, -1, 0))); sharedData[local.x + 1][0][local.z + 1] = uint8_t(data); if (VoxelIsAir(data)) { fullyOpaque = false; }
        data = VoxelIsFilled(VoxelToWorld(global + local.xyz + ivec3(0, 17, 0))); sharedData[local.x + 1][17][local.z + 1] = uint8_t(data); if (VoxelIsAir(data)) { fullyOpaque = false; }

        // X
        data = VoxelIsFilled(VoxelToWorld(global + local.yxz + ivec3(-1, 0, 0))); sharedData[0][local.x + 1][local.z + 1] = uint8_t(data); if (VoxelIsAir(data)) { fullyOpaque = false; }
        data = VoxelIsFilled(VoxelToWorld(global + local.yxz + ivec3(17, 0, 0))); sharedData[17][local.x + 1][local.z + 1] = uint8_t(data); if (VoxelIsAir(data)) { fullyOpaque = false; }

        // Z
        data = VoxelIsFilled(VoxelToWorld(global + local.xzy + ivec3(0, 0, -1))); sharedData[local.x + 1][local.z + 1][0] = uint8_t(data); if (VoxelIsAir(data)) { fullyOpaque = false; }
        data = VoxelIsFilled(VoxelToWorld(global + local.xzy + ivec3(0, 0, 17))); sharedData[local.x + 1][local.z + 1][17] = uint8_t(data); if (VoxelIsAir(data)) { fullyOpaque = false; }
    }

    uint8_t[16] sharedBlockData;

    int sum = 0;
    for (int y = 15; y >= 0; --y) {
        ivec3 tid2 = tid + ivec3(0, y, 0);
        vec3 position = vec3(tid2);
        position = VoxelToWorld(position);

        uint blockData = VoxelIsFilled(position);
        if (blockData != 0) {
            sum += 1;
            filled = true;
        }

        sharedBlockData[y] = uint8_t(blockData);

        sharedData[gl_LocalInvocationID.x + 1][y + 1][gl_LocalInvocationID.z + 1] = uint8_t(blockData);
    }

    if (sum < 16) {
        fullyOpaque = false;
    }

    memoryBarrierShared();
    barrier();

    if ((gl_LocalInvocationID.x) == 0 && (gl_LocalInvocationID.z) == 0)
    {
        int chunkVal = 0;

        if (fullyOpaque) {
            ChunkDataWrite(tid, 2);
            chunkVal = 2;
        }
        else if (filled) {
            ChunkDataWrite(tid, 1);
            chunkVal = 1;
        }
        else {
            ChunkDataWrite(tid, 0);
        }

        ivec4 writeValue = ivec4(-1);
        if (chunkVal == 1) {
            uint id = atomicAdd(bufferFront[IDWORLD], 1);
            ivec3 pos = ivec3(chunkID[DIWORLD(id)]);
            writeValue = ivec4(pos, 0);
        }
        SparseChunkStore(rirCoord(tid) / 16, writeValue);
    }

    memoryBarrierShared();
    barrier();



    for (int y = 15; y >= 0; --y) {
        ivec3 p = ivec3(gl_LocalInvocationID.x, y, gl_LocalInvocationID.z);
        if (true || p.x > 0 && p.x < 15 && p.y > 0 && p.y < 15 && p.z > 0 && p.z < 15) {
            if (
                sharedData[p.x + 1 + 1][p.y + 0 + 1][p.z + 0 + 1] != uint8_t(0) &&
                sharedData[p.x - 1 + 1][p.y + 0 + 1][p.z + 0 + 1] != uint8_t(0) &&
                sharedData[p.x + 0 + 1][p.y + 1 + 1][p.z + 0 + 1] != uint8_t(0) &&
                sharedData[p.x + 0 + 1][p.y - 1 + 1][p.z + 0 + 1] != uint8_t(0) &&
                sharedData[p.x + 0 + 1][p.y + 0 + 1][p.z + 1 + 1] != uint8_t(0) &&
                sharedData[p.x + 0 + 1][p.y + 0 + 1][p.z - 1 + 1] != uint8_t(0)
                ) {
                //sharedBlockData[y] = uint8_t(0);
            }
        }
        else {
            //sharedBlockData[y] = uint8_t(0);
        }
    }

    for (int y = 15; y >= 0; --y) {
        ivec3 tid2 = tid + ivec3(0, y, 0);

        int data = int(sharedBlockData[y]);
        VoxelWrite(ivec3(tid2), data, 0);

        if (data > 0) {
            //    AddVoxel(tid2, data);
        }
    }
};

void main() {
    // MAX_WORLD_COUNT is unrolled along the dispatch .y dimension.
    // This is unlike all the other shaders, which have a world count loop inside the kernel.
    int physicalWorldID = int(gl_WorkGroupID.y);

    ivec3 tid = chunkIndirectCoordinates.data[gl_WorkGroupID.x].xyz*16 + ivec3(gl_LocalInvocationID.xyz);

    if (SetLogicalWorldID(physicalWorldID, (int(VoxelToWorld(tid).z)/16)*16)) {
        main1(tid);
    }
}

#endif
#endif



#ifdef CXX_STAGE
#define DeallocChunks_glsl "Triangulate.glsl", "DEALLOC_CHUNKS_STAGE", "compute"
#endif

#ifdef DEALLOC_CHUNKS_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

uint WarpAtomicAdd() {
    uint liveMask = uint(ballotARB(true));
    uint liveCount = bitCount(liveMask);

    uint prefixSum = bitCount(liveMask & ((1 << gl_SubGroupInvocationARB) - 1));

    uint first_thread = findLSB(liveMask);

    uint vertID = 0;

    if (gl_SubGroupInvocationARB == first_thread) {
        vertID = atomicAdd(bufferBack[IDWORLD], liveCount);
    }

    return readInvocationARB(vertID, first_thread) + int(prefixSum);
}

void main1() {
    ivec3 tid = ivec3(gl_GlobalInvocationID) * ivec3(16);
    
    if (ChunkChanged(tid)) {
        ivec3 pos2 = SparseChunkLoad(rirCoord(tid) / 16);
        if (pos2.x == -1) return;

        //uint id = atomicAdd(bufferBack[IDWORLD], 1u);
        uint id = WarpAtomicAdd();

        chunkID[DIWORLD(id)].rgb = pos2;
        SparseChunkStore(rirCoord(tid) / 16, ivec4(-1));
    }
};

void main() {
    for (int i = 0; i < MAX_WORLD_COUNT; ++i) {
        SetPhysicalWorldID(i); main1();
    }
};
#endif
#endif



#ifdef CXX_STAGE
    #define ClearLod_glsl "Triangulate.glsl", "CLEAR_LOD_STAGE", "compute"
#endif

#ifdef CLEAR_LOD_STAGE
#ifdef COMPUTE_STAGE

layout (local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

void main() {
    int physicalWorldID = int(gl_WorkGroupID.y);
    SetPhysicalWorldID(physicalWorldID);

    ivec3 tid = chunkIndirectCoordinates.data[gl_WorkGroupID.x].xyz * 16 + ivec3(gl_LocalInvocationID.xyz);
    
    // Zero-out the LOD structure
    for (int y = 0; y < 16; y += 1) {
        ivec3 tid2 = tid + ivec3(0, y, 0);
        VoxelWrite(ivec3(tid2), 0, 2);
    }
}
#endif
#endif



#ifdef CXX_STAGE
    #define Topsoil_glsl "Triangulate.glsl", "TOPSOIL_STAGE", "compute"
#endif

#ifdef TOPSOIL_STAGE
#ifdef COMPUTE_STAGE

layout (local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

bool VoxelIsAir(ivec3 pos) {
    if (SPARSE) {
        if (ChunkDataRead(pos) == 2) return false;
    }

    return VoxelRead(pos, 0) == 0;
}

ivec3 rotate(ivec3 c, int i) {
    if (i == 0) return c;
    if (i == 1) return c.zyx * ivec3(1, 1, -1);
    if (i == 2) return c.zyx * ivec3(-1, 1, 1);
    if (i == 3) return c * ivec3(-1, 1, -1);
    return c;
}

void GenerateTree(vec3 position) {
    //return;

    if (distance(VoxelToWorld(position).xy, trackPos) < 20.0) return;

    ivec3 pos = ivec3(position);

    ivec3 pos2 = ivec3(VoxelToWorld(position)) * ivec3(12345, 654321, 1246);

    //float biome = pow(interp(simplex3d_fractal(position / vec3(2048)) * 0.5 + 0.5, 0.2, 0.8), 2.0);
    
    float treeProb = mix(400, 40, interp(simplex3d_fractal(VoxelToWorld(position) / vec3(2048)) * 0.5 + 0.5, 0.2, 0.8));

    if (RandF(pos2.x ^ pos2.y ^ pos2.z) < 1.0/treeProb) {
        VoxelWrite(pos, id_dirt, 0);

        for (int i = 1; i < 7; ++i)
            VoxelWrite(pos + ivec3(0, i, 0), id_oak_log, 0);

        VoxelWrite(pos + ivec3(0, 7, 0), id_leaves, 0);

        for (int y = 4; y <= 5; ++y) {
            for (int i = 0; i < 4; ++i) {
                VoxelWrite(pos + rotate(ivec3(1, y, 0), i), id_leaves, 0);
                VoxelWrite(pos + rotate(ivec3(1, y, 1), i), id_leaves, 0);

                VoxelWrite(pos + rotate(ivec3(2, y, 0), i), id_leaves, 0);
                VoxelWrite(pos + rotate(ivec3(2, y, 2), i), id_leaves, 0);
                VoxelWrite(pos + rotate(ivec3(1, y, 2), i), id_leaves, 0);
                VoxelWrite(pos + rotate(ivec3(2, y, 1), i), id_leaves, 0);
            }
        }

        for (int y = 6; y <= 7; ++y) {
            for (int i = 0; i < 4; ++i) {
                VoxelWrite(pos + rotate(ivec3(1, y, 0), i), id_leaves, 0);
            }
        }

    }
}

void main1(ivec3 tid) {
    vec3 position = vec3(tid);
    ivec3 pos = ivec3(position);

    uint data = VoxelRead(pos, 0);

    if (data == 0)
        return;

    if (data != 1)
        return;

    if (VoxelIsAir(pos + ivec3(0, 1, 0))) {

        if (VoxelToWorld(position).y <= SAND_HEIGHT) {
            VoxelWrite(pos, id_sand, 0);
            return;
        }

        VoxelWrite(pos, id_grass, 0);


        if (ChunkDataRead(pos + ivec3(0, 7, 0)) == 0) return;
        GenerateTree(position);
        
        return;
    }

    if (VoxelToWorld(position).y <= SAND_HEIGHT) {
        for (int i = 2; i <= 4; ++i) if (VoxelIsAir(pos + ivec3(0, i, 0))) { VoxelWrite(pos, id_sand, 0); return; }
    }
        
        else {
        for (int i = 2; i <= 4; ++i) if (VoxelIsAir(pos + ivec3(0, i, 0))) { VoxelWrite(pos, id_dirt, 0); return; }
    }
}

void main2(ivec3 tid) {
    if (!ChunkAllocated(tid)) return;

    for (int y = 15; y >= 0; --y) {
        ivec3 tid2 = tid + ivec3(0,y,0);
        main1(tid2);
    }
};

void main() {
    int physicalWorldID = int(gl_WorkGroupID.y);

    ivec3 tid = chunkIndirectCoordinates.data[gl_WorkGroupID.x].xyz*16 + ivec3(gl_LocalInvocationID.xyz);

    if (SetLogicalWorldID(physicalWorldID, (int(VoxelToWorld(tid).z)/16)*16)) {
        main2(tid);
    }
}

#endif
#endif



#ifdef CXX_STAGE
#define GenerateLOD_glsl "Triangulate.glsl", "GENERATE_LOD_STAGE", "compute"
#endif

#ifdef GENERATE_LOD_STAGE
#ifdef COMPUTE_STAGE

layout(local_size_x = 16, local_size_y = 1, local_size_z = 16) in;

void main1(ivec3 tid) {
    vec3 position = VoxelToWorld(vec3(tid));
    uint data = VoxelRead(tid, 0);

    if (data > 0) {
        VoxelWrite(ivec3(tid), 1, 2);
    }
};

void main2(ivec3 tid) {
    if (!ChunkAllocated(tid)) return;

    for (int y = 0; y < 16; y += 1) {
        ivec3 tid2 = tid + ivec3(0, y, 0);
        main1(tid2);
    }
};

void main() {
    int physicalWorldID = int(gl_WorkGroupID.y);

    ivec3 tid = chunkIndirectCoordinates.data[gl_WorkGroupID.x].xyz*16 + ivec3(gl_LocalInvocationID.xyz);

    if (SetLogicalWorldID(physicalWorldID, (int(VoxelToWorld(tid).z)/16)*16)) {
        main2(tid);
    }
}

#endif
#endif
