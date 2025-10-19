
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
    float nearestBeat = beatsSSBO[BinarySearchNearest(int(p.z))].zPos;
    float distToBeat = length((vec3(trackPos, nearestBeat) - p) / vec3(2.0,1.0,2.0));

    float curr = 0.0;

    curr += interp(distToBeat, 32.0, 0.0);


    {
        curr += interp(p.z, GetBeatPos(670 - 1), GetBeatPos(670)) * interp(p.y-trackPos.y, -10, 0) * 0.3;
    }

    return clamp(curr, 0.0, 1.0);

    //return 0.0;
    return mix((1 - interp(distance(p.xy, trackPos), 2, 32)) * (1 - interp(p.y - trackPos.y, 10, -20)), 0.0, 0.5);
}

float SculptAdd(vec3 p) {
    return 0.0;
    //return interp(distance(p.xy, trackPos + 100), 32.0, 0.0);
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
    vec2 sel2 = vec2(1.0 - sel, sel);
    sel2.x *= interp(p.y, 128, WATER_HEIGHT);
    sel2.y *= interp(p.y, 192, WATER_HEIGHT);
    //sel2 = mix(sel2, vec2(0), SculptRemove(p));
    //sel2 = mix(sel2, vec2(1), SculptAdd(p));
    float ret = 0.0;
    if (sel2.x > 0.0) ret += sel2.x * Simplex(p, vec3(256));
    if (sel2.y > 0.0) ret += sel2.y * Simplex(p, vec3(171), vec3(1e3));
    return ret;
}

float BetaTerrain2(vec3 p, float water_height) {
    float sel = interp(Simplex(p, vec3(1024, 1e8, 1024), vec3(0)), 0.45, 0.55);
    vec2 sel2 = vec2(1.0 - sel, sel);
    //sel2 = mix(sel2, vec2(0), SculptRemove(p));
    //sel2 = mix(sel2, vec2(1), SculptAdd(p));
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

float TerrainBoolean2(vec3 p) {
    vec3 position=p;

    if (IS_WORLD_NAME(0)) {
        return
        mix(
            p.y<trackPos.y+40 ? 0.85*Simplex(p+vec3(0,0,0), vec3(1e16, 16, 1e16), vec3(0), vec3(0, 0, 0), vec3(4, 10, 4))
                                : 0.0,
            0.85*Simplex(p, vec3(256)),
            0.5)
            //*mix(0.7,1.0, interp(distance(p.z,GetBeatPos(28.2)+5),0,10))
            *mix(0.0,1.0, interp(distance(p.xz,vec2(trackPos.x,GetBeatPos(49)+50)),0,100))
            //*mix(0.5,1.0,interp(length(trackDist),0,10))
            *mix(0.8,1.0,interp(trackDist.x,0,10))
            //*mix(0.5,1.0,interp(trackDist.y,0,10))
            ;
    }

    if (IS_WORLD_NAME(0)) return BetaTerrain(p);

    if (IS_WORLD_NAME(1)) return TheCave(p);
    
    if (IS_WORLD_NAME(3))  {
        vec3 p2 = p;
        {
            p2.xy -= trackPos.xy;
            p2.xy = rotate(floor(p2.z/100.0 * 5.0) / 5.0) * p2.xy;
            p2.xy += trackPos.xy;
        }

        float v = Simplex(p2, vec3(171) * vec3(6,1,1));
        
        v = mix(v, 0.0, interp(length(trackDist.xy), 0.0, 1000.0));
        v = mix(v, 0.0, 0.5 * interp(-(p2.x - trackPos.x), 100.0, 0.0));
        //if (length(trackDist.xy) > 100.0) return 0.0;
        //v = mix(v, Simplex(p, vec3(256)), interp(p.y, 50, 10));
        //v = mix(v, 1.0, 0.5*interp(p.y, 10.0, 0));
        return v;
    }

    if (IS_WORLD_NAME(3)) return BetaTerrain(p);

    if (p.z < GetBeatPos(313)) return FarLands_Edge_V3style(p);

    return BetaTerrain(p);
}

float SculptRemove2(vec3 p) { vec3 position = p;
    float curr = 0.0;
    //float distToBeat = length((vec3(trackPos, nearestBeat) - p) / vec3(2.0, 1.0, 2.0));
    ///curr += interp(distToBeat, 32.0, 0.0);

    float flattener = interp(p.y - trackPos.y, -10, 0) * 0.3;
    float sunrise = interp(p.z, GetBeatPos(670 - 1), GetBeatPos(670));
    sunrise *= interp(p.z, GetBeatPos(700), GetBeatPos(697));
    curr += flattener * sunrise;

    curr += interp(length(trackDist.xy), 20.0, 5.0);
    return clamp(curr, 0.0, 1.0);
}

float SculptAdd2(vec3 p) { vec3 position = p;
    //return interp(length(trackDist - vec2(40.0)), 40.0, 0.0);
    
    float nearestBeat = beatsSSBO[BinarySearchNearest(int(p.z))].zPos;
    float distToBeat = length((vec3(trackPos-vec2(10,0), nearestBeat) - p) / vec3(1.0, 1.0, 1.0));
    //return mix(0.0, interp(distToBeat, 10.0, 0.0), 0.8);
    
    return 0.0;
}

bool TerrainBoolean3(vec3 p) { vec3 position=p;
    float noise = TerrainBoolean2(p);

    noise = mix(noise, 0.0, SculptRemove2(p));
    noise = mix(noise, 1.0, SculptAdd2(p));

    return noise > 0.5;
}

bool TerrainBoolean(vec3 p) { vec3 position=p;
    if (p.z < GetBeatPos(721)) return TerrainBoolean3(p);
    
    return TerrainBoolean3(p);

    // Example of how to do iterated terrain
    for (int i = 0; i < 4; ++i) {
        if (TerrainBoolean3(p + vec3(0.0, 0.0, i*40.0)))
            return true;
    }

    return false;
}

float FUNCTION_0(vec3 p) {
//#include "Worlds.glsl"
}

float FUNCTION_1(vec3 p) {
#define Simplex(x, y, z) 1.0
//#include "Worlds.glsl"
#undef Simplex
}

uint TerrainAndWater(vec3 p) {
    bool terrain = FUNCTION_1(p) > 0.5;
    if (terrain) {
        terrain = FUNCTION_0(p) > 0.5;
    }

    if (int(VoxelToWorld(p).y) == GetWaterHeight() && !terrain) return id_water;

    return terrain ? id_stone : 0;
}

uint VoxelIsFilled(vec3 position) { vec3 p = position;
    if (position.y >= WORLD_SIZE.y - 10) return 0;
    if (int(VoxelToWorld(position).y) < GetWaterHeight()) return 0;

    if (beatFromPos < 97) return TerrainAndWater(position);

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
