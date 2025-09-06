
#if !defined CXX_STAGE

#define IDWORLD 0
#define DIWORLD(id) (id % sparseTotalSize)

#define trackDist (abs(position.xy - (trackPos.xy + vec2(0, 1)) + 0.5))

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

#define id_both 0
#define id_left 1
#define id_right 2
bool CheckExtent(vec3 position, int x0, int x1, int y0, int y1, int leftright) {
    ivec2 ebin = ivec2(position.xy) - ivec2(trackPos.xy);

    if (leftright == id_left) ebin.x = -ebin.x;

    if (leftright == id_both) ebin.x = abs(ebin.x);

    return ebin.x >= x0 && ebin.x <= x1 && ebin.y >= y0 && ebin.y <= y1;
}

int CheckPosition(vec3 position, int idx, bool exact) {
    int beatType = BEAT_TYPE(idx);

    if (beatType == beat_type_portal) return 0;

    if (beatType == beat_type_programmatic) {
        int height = (2 + idx) % 4;
        if (exact && CheckExtent(position, 2, 2, height+1, height+1, id_both)) return id_torch;
        
        if (CheckExtent(position, 2, 2, height, height, id_both)) return id_permastone;
        return 0;
    }

    if (!exact) return 0;

    int beatSide = beatType >= 20 ? id_both : (beatType >= 10 ? id_left : (id_right));
    beatType %= 10;
    bool blockRightSide = position.x - trackPos.x > 0.0;

    if (beatType == r_low) {
        if (CheckExtent(position, 1, 1, 0, 0, beatSide)) return id_permastone;
        if (CheckExtent(position, 1, 1, 1, 1, beatSide)) return id_torch;
    } else if (beatType == r_default) {
        if (CheckExtent(position, 1, 1, 0, 1, beatSide)) return id_permastone;
        if (CheckExtent(position, 1, 1, 2, 2, beatSide)) return id_torch;
    } else if (beatType == r_wide) {
        if (CheckExtent(position, 2, 2, 2, 2, beatSide)) return id_permastone;
        if (CheckExtent(position, 1, 1, 2, 2, beatSide)) return blockRightSide ? id_torch_right : id_torch_left;
        //if (CheckPosition(position - vec3(0, 1, 0), idx3)>0) return id_torch_right;
    } else {
        if (CheckExtent(position, 1, 1, 0, 1, id_both)) return id_permastone;
        if (CheckExtent(position, 1, 1, 2, 2, id_both)) return id_torch;
    }

    return 0;
}

const float Lacunarity = 2;
const float Persistence = 1.0/Lacunarity;

float NewSimplex2(vec3 m) {
    int num_octaves = 8;
    float sum = 0.0;
    float noise = 0.0;
    float frequency = 1.0;  // Starting relative frequency
    float amplitude = 1.0;  // Starting relative amplitude
    for (int i = 0; i < num_octaves; ++i) {
        noise += (simplex3d(m * frequency)*0.5+0.5) * amplitude;
        sum += amplitude;
        frequency *= Lacunarity;
        amplitude *= Persistence;

    }

    return noise / sum;
}

const float bene = 12550820.0;

float NewSimplex4(vec3 m, vec3 scale, vec3 offset) {
    m = m / scale + offset;
    return NewSimplex2(m);
}

float Simplex(vec3 m, vec3 scale, vec3 offset, vec3 trilinearOffset, vec3 tilinearScale) {
    //if (int(m.z) % 1000 < 500) return 0;

    if (!TRILINEAR_TERRAIN) return NewSimplex4(m, scale, offset);

    m += trilinearOffset;

    vec3 size = tilinearScale;
    vec3 basecoord = floor(m / size)*size;

    vec3 interpol = (m-basecoord)/size;

    float x1 = mix(NewSimplex4(basecoord+size*vec3(0,0,0),scale,offset), NewSimplex4(basecoord+size*vec3(1,0,0), scale, offset), interpol.x);
    float x2 = mix(NewSimplex4(basecoord+size*vec3(0,0,1),scale,offset), NewSimplex4(basecoord+size*vec3(1,0,1), scale, offset), interpol.x);

    float x3 = mix(NewSimplex4(basecoord+size*vec3(0,1,0),scale,offset), NewSimplex4(basecoord+size*vec3(1,1,0), scale, offset), interpol.x);
    float x4 = mix(NewSimplex4(basecoord+size*vec3(0,1,1),scale,offset), NewSimplex4(basecoord+size*vec3(1,1,1), scale, offset), interpol.x);
    return mix(mix(x1, x2, interpol.z), mix(x3, x4, interpol.z), interpol.y);
}

float Simplex(vec3 m, vec3 scale, vec3 offset, vec3 trilinearOffset) {
    return Simplex(m, scale, offset, trilinearOffset, vec3(4, 10, 4));
}

float Simplex(vec3 m, vec3 scale, vec3 offset) {
    return Simplex(m, scale, offset, vec3(0));
}

float Simplex(vec3 m, vec3 scale) {
    return Simplex(m, scale, vec3(0), vec3(0));
}






float SculptRemove(vec3 p) {
    return 1.0;
    return mix(1 - (1 - interp(distance(p.xy, trackPos), 2, 32)) * (1 - interp(p.y - trackPos.y, 10, -20)), 1.0, 0.5);
}

float SculptAdd(vec3 p) {
    return 1.0;
    return interp(distance(p.xy, trackPos + 100), 2.0, 32.0);
}


float FarLands_Edge1(vec3 position) { vec3 p = position;
    return
    mix(
        Simplex(p, vec3(16, 16, 1e16), vec3(0,-12.55082,0)),
        Simplex(p, vec3(16, 16, 1e16), vec3(0, 12.55082,0)),
        interp(Simplex(p, vec3(256, 1e35, 256)), 0.45, 0.55));
}

float FarLands_Edge_V3style(vec3 p) {
    if (p.y > trackPos.y) return 0.0;
    return
    mix(
        Simplex(p, vec3(16, 16, 1e16), vec3(0,-12.55082,0)),
        Simplex(p, vec3(16, 16, 1e16), vec3(0, 12.55082,0)),
        interp(Simplex(p, vec3(256, 1e35, 256)), 0.45, 0.55));
}

float FarLands_Edge_V2tunnel(vec3 p) {
    if (any(greaterThan(abs(p.xy-trackPos), vec2(20, 30)))) return 0.0;
    p.x -= 800.0 - 5;
    //p.y += 18;
    //if (p.y > trackPos.y+16) return 0.0;
    return
        mix(
            Simplex(p, vec3(16, 16, 1e16), vec3(0, -12.55082, 0)),
            Simplex(p, vec3(16, 16, 1e16), vec3(0, 12.55082, 0)),
            interp(Simplex(p, vec3(256, 1e35, 256)), 0.45, 0.55));
}

float FarLands_Corner1(vec3 position) { vec3 p = position;
    return
    mix(
        Simplex(p, vec3(1e16, 16, 1e16), vec3(0), vec3(0,0,0), vec3(4,10,4)),
        Simplex(p, vec3(1e16, 16, 1e16), vec3(1e3), vec3(0,36,0), vec3(4,10.1,4)),
        interp(Simplex(p, vec3(256, 1e16, 256)), 0.45, 0.55));
}

float FarLands_Corner2(vec3 position) {
    vec3 p = position;
    return
        mix(
            Simplex(p, vec3(1e16, 16, 1e16), vec3(0), vec3(0, 0, 0), vec3(4, 10, 4)),
            Simplex(p, vec3(1e16, 16, 1e16), vec3(1e3), vec3(0, 20, 0), vec3(4, 16, 4)),
            interp(Simplex(p, vec3(256, 1e16, 256)), 0.45, 0.55));
}

float BetaTerrain(vec3 p) {
    float sel = interp(Simplex(p, vec3(1024, 1e8, 1024), vec3(0)), 0.45, 0.55);
    float height = mix(128, 128, pow(Simplex(p, vec3(2048, 1e8, 2048), vec3(1e5)), 2.0));
    vec2 sel2 = vec2(1.0 - sel, sel);
    sel2.x *= interp(p.y, 128, WATER_HEIGHT);
    sel2.y *= interp(p.y, 192, WATER_HEIGHT);
    sel2 = mix(vec2(1), sel2 * SculptRemove(p), SculptAdd(p));
    float ret = 0.0;
    if (sel2.x > 0.0) ret += sel2.x * Simplex(p, vec3(256));
    if (sel2.y > 0.0) ret += sel2.y * Simplex(p, vec3(171), vec3(1e3));
    return ret;
}

bool NewTerrain(vec3 position) { vec3 p = position;
    if (position.y > 256) return false;
    
    if (false&&p.z < GetBeatPos(313))
        { return FarLands_Edge_V3style(p) > 0.5; }

    //if (false)
    {
        return BetaTerrain(p) > 0.5;
    }

    return
        mix(
            Simplex(p, vec3(256)) * interp(p.y, 256, WATER_HEIGHT),
            Simplex(p, vec3(256), vec3(1e3)) * interp(p.y, 256, WATER_HEIGHT),
            interp(Simplex(p, vec3(2048, 1e35, 2048), vec3(0)), 0.5, 0.52)
        ) > 0.5;

    return
    mix(
        Simplex(p, vec3(256), vec3(0)) * interp(p.y, 256, WATER_HEIGHT),
        mix(
            FarLands_Corner2(p),
            FarLands_Edge1(p),
            interp(p.x-trackPos.x, -10.0, 10.0)),
        0*interp(Simplex(p, vec3(2048, 1e35, 2048), vec3(0)), 0.5, 0.52)
        +interp(distance(p.x, trackPos.x), 0.0, 200.0))
            > 0.5;
}

float Pre_Chorus1(vec3 p) {
    float ret = 0.0;
    float sel = interp(p.z, GetBeatPos(313 - 6), GetBeatPos(313));
    if (sel < 1.0) ret += FarLands_Edge_V2tunnel(p) * (1-sel);
    if (sel > 0.0) ret += BetaTerrain(p) * sel;
    return ret;
}

float TheCave(vec3 p) {
    if (p.y > trackPos.y + 20.0) return 1.0;

    return 0.0;
}

#define IS_WORLD_NAME(n) (g_logicalWorldID == WORLD_NAME(n))

bool TerrainBoolean(vec3 p) {
    //return BetaTerrain(p) > 0.5;

    if (IS_WORLD_NAME(0)) return BetaTerrain(p) > 0.5;

    if (IS_WORLD_NAME(1)) return TheCave(p) > 0.5;
    
    if (IS_WORLD_NAME(5)) return Pre_Chorus1(p) > 0.5;

    if (p.z < GetBeatPos(313)) return FarLands_Edge_V3style(p) > 0.5;

    return NewTerrain(p);
}

uint GetWaterHeight(vec3 p) {
    //if (IS_WORLD_NAME(5)) return uint(trackPos.y-1);

    return WATER_HEIGHT;
}

uint TerrainAndWater(vec3 p) {
    bool terrain = TerrainBoolean(p);
    
    if (int(VoxelToWorld(p).y) == GetWaterHeight(p) && !terrain) return id_water;

    return terrain ? id_stone : 0;
}






uint VoxelIsFilled(vec3 position) { vec3 p = position;
    if (position.y >= WORLD_SIZE.y - 10) return 0;
    if (int(VoxelToWorld(position).y) < GetWaterHeight(p)) return 0;

    int idx = BinarySearchGT(int(position.z));
    bool exact = BinarySearchIsExact(int(position.z), idx);
    int beatType = BEAT_TYPE(idx);

    if (beatType == beat_type_portal) {
        int portalPos = int(beatsSSBO[idx].zPos);
        int portalDist = int(position.z) - portalPos;
        if (g_logicalWorldID != PORTAL_TARGET(idx)) {
            if (portalDist == 1+2 && MediumAirTunnel2()) return id_portal_forward;
            if (portalDist == 0+2 && MediumAirTunnelBorder2()) return id_permastone;
            if (portalDist == 1+2 && MediumAirTunnelBorder2()) return id_permastone;
            if (portalDist == 2+2 && (MediumAirTunnel2() || MediumAirTunnelBorder2())) return id_permastone;
        }
        else {
            if (portalDist == -1+2 && MediumAirTunnel2()) return id_portal_backward;
            if (portalDist ==  0+2 && MediumAirTunnelBorder2()) return id_permastone;
            if (portalDist == -1+2 && MediumAirTunnelBorder2()) return id_permastone;
            if (portalDist == -2+2 && (MediumAirTunnel2() || MediumAirTunnelBorder2())) return id_permastone;
        }
    }

    // Filter everything outside the big circle
    if (distance(position.xy, cPos) < mix(5.0, 12.0, interp(position.z, GetCameraPos(265).z, GetCameraPos(313).z))) {
    if (SmallestAirTunnel()) {
        return 0;
    }

    // The cobblestone track
    if (int(position.x) == int(trackPos.x) && int(position.y) == int(trackPos.y)) {
        if (idx < 0 || torchSection(idx)) {
            return id_permastone;
        }
    }

    // Torch beats
    if (torchSection(idx)) {
        int cobble = CheckPosition(position, idx, exact);
        if (cobble > 0) return cobble;
    }

    if (MediumAirTunnel()) return 0;

    if (idx >= 0 &&
        !torchSection(idx) &&
        !bool(trackDist.x < beatRadius + 2 && trackDist.y < beatRadius + 2)) {
        vec3 crunched = crunch(position, vec3(1, 1, freq));
        crunched.y += idx * 8.0;
        float value = (simplex3d_fractal(crunched * vec3(1, 1, 0) / 16.0 / vec3(1, 0.25, 1)));
        if (value > 0.4) return exact ? id_beat : id_stone2;
    }

    if (position.z > GetBeatPos(311.0) && position.z < GetBeatPos(361.0)) return 0;
    }

    return TerrainAndWater(position);
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
        if (UPDATE_INDIRECT) {
            computeIndirect.num_groups[chunkUpdates].x = 0;
            computeIndirect.num_groups[chunkUpdates].y = 1;
            computeIndirect.num_groups[chunkUpdates].z = 1;
        } else {
            computeIndirect.num_groups[chunkUpdates].x = int(WORLD_SIZE.x / 16);
            computeIndirect.num_groups[chunkUpdates].y = int(WORLD_SIZE.y / 16);
            computeIndirect.num_groups[chunkUpdates].z = int(WORLD_SIZE.z / 16);
        }
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
    if (!UPDATE_INDIRECT) return;

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
    ivec3 tid = ivec3(gl_GlobalInvocationID) * ivec3(1, 16, 1);

    if (UPDATE_INDIRECT) {
        tid = chunkIndirectCoordinates.data[gl_GlobalInvocationID.x / 16].xyz * 16 + ivec3(gl_GlobalInvocationID.xyz % 16);
    } else {
        if (!ChunkChanged(tid)) return;
    }

    for (int i = 0; i < MAX_WORLD_COUNT; ++i) {
        if (SetLogicalWorldID(i, (int(VoxelToWorld(tid).z)/16)*16)) {
            main1(tid);
        }
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

void main1(ivec3 tid) {
    VoxelWrite(ivec3(tid), 0, 2);
};

// Zero-out the LOD structure
void main2() {
    ivec3 tid = ivec3(gl_GlobalInvocationID) * ivec3(1, 16, 1);

    if (UPDATE_INDIRECT) {
        tid = chunkIndirectCoordinates.data[gl_GlobalInvocationID.x / 16].xyz * 16 + ivec3(gl_GlobalInvocationID.xyz % 16);
    } else {
        if (!ChunkChanged(tid)) return;
    }

    for (int y = 0; y < 16; y += 1) {
        ivec3 tid2 = tid + ivec3(0, y, 0);
        main1(tid2);
    }
};

void main() {
    for (int i = 0; i < MAX_WORLD_COUNT; ++i) {
        SetPhysicalWorldID(i); main2();
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
    ivec3 tid = ivec3(gl_GlobalInvocationID) * ivec3(1, 16, 1);

    if (UPDATE_INDIRECT) {
        tid = chunkIndirectCoordinates.data[gl_GlobalInvocationID.x / 16].xyz * 16 + ivec3(gl_GlobalInvocationID.xyz % 16);
    } else {
        if (!ChunkChanged(tid)) return;
    }

    for (int i = 0; i < MAX_WORLD_COUNT; ++i) {
        if (SetLogicalWorldID(i, (int(VoxelToWorld(tid).z)/16)*16)) {
            main2(tid);
        }
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
    ivec3 tid = ivec3(gl_GlobalInvocationID) * ivec3(1, 16, 1);

    if (UPDATE_INDIRECT) {
        tid = chunkIndirectCoordinates.data[gl_GlobalInvocationID.x / 16].xyz * 16 + ivec3(gl_GlobalInvocationID.xyz % ivec3(16));
    }

    for (int i = 0; i < MAX_WORLD_COUNT; ++i) {
        if (SetLogicalWorldID(i, (int(VoxelToWorld(tid).z)/16)*16)) {
            main2(tid);
        }
    }
}

#endif
#endif
