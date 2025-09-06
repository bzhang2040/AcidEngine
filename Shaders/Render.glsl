#if !defined(CXX_STAGE)

int GetFaceIndex(vec3 normal) {
    if (normal.z < -0.9) return 0; //vec3(0, 0, -1); }
    if (normal.x < -0.9) return 1; //vec3(-1, 0, 0); }
    if (normal.y < -0.9) return 2; //vec3(0, -1, 0); }
    if (normal.z >  0.9) return 3; //vec3(0, 0, 1); }
    if (normal.x >  0.9) return 4; //vec3(1, 0, 0); }
    if (normal.y >  0.9) return 5; //vec3(0, 1, 0); }
    return 0;
}

#define TIME 0.0
#define WAVE_SPEED 1.0
#define WAVE_MULT 1.0


float GetWaveCoord(float coord) {
    const float madd = 0.5 * noiseResInverse;
    float whole = floor(coord);
    coord = whole + cubesmooth(coord - whole);

    return coord * noiseResInverse + madd;
}

vec2 GetWaveCoord(vec2 coord) {
    const vec2 madd = vec2(0.5 * noiseResInverse);
    vec2 whole = floor(coord);
    coord = whole + cubesmooth(coord - whole);

    return coord * noiseResInverse + madd;
}

float SharpenWave(float wave) {
    wave = 1.0 - abs(wave * 2.0 - 1.0);

    return wave < 0.78 ? wave : (wave * -2.5 + 5.0) * wave - 1.6;
}

const vec4 heights = vec4(29.0, 15.0, 17.0, 4.0);
const vec4 height = heights * WAVE_MULT / (heights.x + heights.y + heights.z + heights.w);

const vec2[4] scale = vec2[4](
    vec2(0.0065, 0.0052)* noiseRes* noiseScale,
    vec2(0.013, 0.00975)* noiseRes* noiseScale,
    vec2(0.0195, 0.014625)* noiseRes* noiseScale,
    vec2(0.0585, 0.04095)* noiseRes* noiseScale);

const vec4 stretch = vec4(
    scale[0].x * -1.7,
    scale[1].x * -1.7,
    scale[2].x * 1.1,
    scale[3].x * -1.05);

const vec2 disp1 = vec2(0.04155, -0.0165) * noiseRes * noiseScale;
const vec2 disp2 = vec2(0.017, -0.0469) * noiseRes * noiseScale;
const vec2 disp3 = vec2(0.0555, 0.03405) * noiseRes * noiseScale;
const vec2 disp4 = vec2(0.00825, -0.0491625) * noiseRes * noiseScale;


const float wt = TIME * WAVE_SPEED * 0.6;
const mat4x2 waveTime = mat4x2(wt * disp1, wt * disp2, wt * disp3, wt * disp4);

float GetWaves(vec2 coord, inout mat4x2 c) {
    float waves = 0.0;
    vec2 ebin;

    c[0].xy = coord * scale[0] + waveTime[0];
    c[0].y = coord.x * stretch[0] + c[0].y;
    ebin = GetWaveCoord(c[0].xy);
    c[0].x = ebin.x;

    waves += SharpenWave(texture(noisetex, ebin).x) * height.x;

    c[1].xy = coord * scale[1] + waveTime[1];
    c[1].y = coord.x * stretch[1] + c[1].y;
    ebin = GetWaveCoord(c[1].xy);
    c[1].x = ebin.x;

    waves += texture(noisetex, ebin).x * height.y;

    c[2].xy = coord * scale[2] + waveTime[2];
    c[2].y = coord.x * stretch[2] + c[2].y;
    ebin = GetWaveCoord(c[2].xy);
    c[2].x = ebin.x;

    waves += texture(noisetex, ebin).x * height.z;

    c[3].xy = coord * scale[3] + waveTime[3];
    c[3].y = coord.x * stretch[3] + c[3].y;
    ebin = GetWaveCoord(c[3].xy);
    c[3].x = ebin.x;

    waves += texture(noisetex, ebin).x * height.w;

    return waves;
}

float GetWaves(vec2 coord) {
    mat4x2 c;

    return GetWaves(coord, c);
}

float GetWaves(mat4x2 c, float offset) {
    float waves = 0.0;

    c[0].y = GetWaveCoord(offset * scale[0].y + c[0].y);

    waves += SharpenWave(texture(noisetex, c[0].xy).x) * height.x;

    c[1].y = GetWaveCoord(offset * scale[1].y + c[1].y);

    waves += texture(noisetex, c[1].xy).x * height.y;

    c[2].y = GetWaveCoord(offset * scale[2].y + c[2].y);

    waves += texture(noisetex, c[2].xy).x * height.z;

    c[3].y = GetWaveCoord(offset * scale[3].y + c[3].y);

    waves += texture(noisetex, c[3].xy).x * height.w;

    return waves;
}

vec2 GetWaveDifferentials(vec2 coord, float scale) { // Get finite wave differentials for the world-space X and Z coordinates
    mat4x2 c;

    float a = GetWaves(coord, c);
    float aX = GetWaves(coord + vec2(scale, 0.0));
    float aY = GetWaves(c, scale);

    return a - vec2(aX, aY);
}

vec3 ComputeWaveNormals(vec3 worldSpacePosition, vec3 flatWorldNormal) {
    //if (WAVE_MULT == 0.0) return vec3(0.0, 0.0, 1.0);

    float angleCoeff = dot(normalize(-worldSpacePosition.xyz), normalize(flatWorldNormal));
    angleCoeff /= clamp(length(worldSpacePosition) * 0.05, 1.0, 10.0);
    angleCoeff = clamp(angleCoeff * 2.5, 0.0, 1.0);
    angleCoeff = sqrt(angleCoeff);
    vec3 worldPos = worldSpacePosition + cameraPosition;
    worldPos.xz = worldPos.xz + worldPos.y;

    vec2 diff = GetWaveDifferentials(worldPos.xz, 0.1) * angleCoeff;
    //return vec3(angleCoeff, 0, 0);
    return vec3(diff, sqrt(1.0 - dot(diff, diff)));
}

//#include SKY

#endif


#ifdef CXX_STAGE
    #define Composite0_glsl "Render.glsl", "COMPOSITE0_STAGE", "compute"
#endif

#ifdef COMPOSITE0_STAGE

#if defined(COMPUTE_STAGE)

vec4 OutColor = vec4(0.0);
vec4 OutColor1 = vec4(0.0);

layout(binding = 4, rgba32f) uniform image2D OutColor0Image;
layout(binding = 5, rgba32f) uniform image2D OutColor1Image;

layout(local_size_x = 16, local_size_y = 16) in;

#define uv vec2((gl_GlobalInvocationID.xy + 0.5) / viewSize.xy)
#define gl_FragCoord vec2(gl_GlobalInvocationID + 0.5)

#if !writeFrames
vec4 debug = vec4(0.0);
void show(bool x) { debug = vec4(vec3(x), 1.0); }
void show(float x) { debug = vec4(x, x, x, 1.0); }
void show(vec2 x) { debug = vec4(x.rg, 0.0, 1.0); }
void show(vec3 x) { debug = vec4(x.rgb, 1.0); }
void show(vec4 x) { debug = vec4(x.rgb, 1.0); }
#define showa(x) show(abs(x))
#define shown(x) show(-(x))
#define shownn(x) show(clamp((x), 0.0, 1.0) + clamp(-(x), 0.0, 1.0))
#define show(x) show(x);
void exit() {
    if (debug.a > 0.0) OutColor = debug * vec4(1,1,1,-1);
}
#else
#define show(x)
#define exit()
#endif

mat3 RecoverTangentMat(vec3 plane) {
    mat3 tbn;

    vec3 plane3 = abs(plane);

    tbn[0].z = -plane.x;
    tbn[0].y = 0.0;
    tbn[0].x = plane3.y + plane.z;

    tbn[1].x = 0.0;
    tbn[1].y = -plane3.x - plane3.z;
    tbn[1].z = plane3.y;

    tbn[0] *= -1.0;
    tbn[1] *= -1.0;

    tbn[2] = plane;

    if (plane.y < -0.5) tbn = mat3(1, 0, 0, 0, 0, -1, 0, -1, 0);

    return tbn;
}

float linearizeDepth(float depth, float near, float far) {
    float z_n = 2.0 * depth - 1.0; // Transform [0, 1] to [-1, 1]
    return 2.0 * near * far / (far + near - z_n * (far - near));
}

vec3 fMin(vec3 a) {
    // Returns a unit vec3 denoting the minimum element of the parameter.
    // Example:
    // fMin( vec3(1.0, -2.0, 3.0) ) = vec3(0.0, 1.0, 0.0)
    // fMin( vec3(0.0,  0.0, 0.0) ) = vec3(0.0, 0.0, 1.0) <- defaults to Z

    vec2 b = clamp(clamp((a.yz - a.xy), 0.0, 1.0) * (a.zx - a.xy) * 1e35, 0.0, 1.0);
    return vec3(b.x, b.y, 1.0 - b.x - b.y);

    // Alternate version	
    // Note: this handles the situation where they're all equal differently
    // return vec3(lessThan(a.xyz, a.yzx) && lessThan(a.xyz, a.zxy));
}


float fMin(vec3 a, out vec3 val) {
    float ret = min(a.x, min(a.y, a.z));
    vec2 c = 1.0 - clamp((a.xy - ret) * 1e35, 0.0, 1.0);
    val = vec3(c.xy, 1.0 - c.x - c.y);
    return ret;
}

vec3 fMax(vec3 a) {
    if (a.x >= a.y && a.x >= a.z) return vec3(1, 0, 0);
    if (a.y >= a.x && a.y >= a.z) return vec3(0, 1, 0);
    return vec3(0, 0, 1);
}

struct VoxelIntersectOut {
    bool hit;
    vec3 voxelPos;
    vec3 plane;
};

#define BinaryDot(a, b) ((a.x & b.x) | (a.y & b.y) | (a.z & b.z))
#define BinaryMix(a, b, c) ((a & (~c)) | (b & c))

float BinaryDotF(vec3 v, ivec3 uplane) {
    ivec3 u = floatBitsToInt(v);
    return intBitsToFloat(BinaryDot(u, uplane));
}

float MinComp(vec3 v, out vec3 minCompMask) {
    float minComp = min(v.x, min(v.y, v.z));
    minCompMask.xy = 1.0 - clamp((v.xy - minComp) * 1e35, 0.0, 1.0);
    minCompMask.z = 1.0 - minCompMask.x - minCompMask.y;
    return minComp;
}

ivec3 GetMinCompMask(vec3 v) {
    ivec3 ia = floatBitsToInt(v);
    ivec3 iCompMask;
    iCompMask.xy = ((ia.xy - ia.yx) & (ia.xy - ia.zz)) >> 31;
    iCompMask.z = (-1) ^ iCompMask.x ^ iCompMask.y;

    return iCompMask;
}

ivec2 GetNonMinComps(ivec3 xyz, ivec3 uplane) {
    return BinaryMix(xyz.xz, xyz.yy, uplane.xz);
}

int GetMinComp(ivec3 xyz, ivec3 uplane) {
    return BinaryDot(xyz, uplane);
}

ivec3 SortMinComp(ivec3 xyz, ivec3 uplane) {
    ivec3 ret;
    ret.xy = GetNonMinComps(xyz, uplane);
    ret.z = (xyz.x ^ xyz.y) ^ xyz.z ^ (ret.x ^ ret.y);
    return ret;
}

ivec3 UnsortMinComp(ivec3 uvw, ivec3 uplane) {
    ivec3 ret;
    ret.xz = BinaryMix(uvw.xy, uvw.zz, uplane.xz);
    ret.y = (uvw.x ^ uvw.y) ^ uvw.z ^ (ret.x ^ ret.z);
    return ret;
}

bool OutOfVoxelBounds(int point, ivec3 uplane) {
    int comp = (int(WORLD_SIZE.x) & uplane.x) | (int(WORLD_SIZE.y) & uplane.y) | (int(WORLD_SIZE.z) & uplane.z);
    return point >= comp || point < 0;
}

struct Hit2 { float tmin; float tmax; vec3 normal; };
struct Ray2 { vec3 origin; vec3 dir; vec3 invDir; };
bool BBoxIntersect(const vec3 boxMin, const vec3 boxMax, const Ray2 r, out Hit2 hit) {
    vec3 tbot = r.invDir * (boxMin - r.origin);
    vec3 ttop = r.invDir * (boxMax - r.origin);

    vec3 tmin3 = min(ttop, tbot);
    vec3 tmax3 = max(ttop, tbot);

    float t0 = max(max(tmin3.x, tmin3.y), tmin3.z);
    float t1 = min(min(tmax3.x, tmax3.y), tmax3.z);

    hit.tmin = t0;
    hit.tmax = t1;

    if (t1 > max(t0, 0.0)) {
        // Find the axis that contributed to t0
        if (t0 == tmin3.x) {
            hit.normal = vec3(t0 == ttop.x ? 1.0 : -1.0, 0.0, 0.0);
        }
        else if (t0 == tmin3.y) {
            hit.normal = vec3(0.0, t0 == ttop.y ? 1.0 : -1.0, 0.0);
        }
        else {
            hit.normal = vec3(0.0, 0.0, t0 == ttop.z ? 1.0 : -1.0);
        }
        return true;
    }
    return false;
}
bool SubVoxelTrace = true;

mat3 Mat3XZ(mat2 mat) {
    return mat3(
        mat[0].x, 0.0, mat[0].y,  // The first column is formed by swapping x and z.
        0.0, 1.0, 0.0,  // The second column is the same as the input matrix.
        mat[1].x, 0.0, mat[1].y            // The third column is the z-axis, which is just [0, 0, 1].
    );
}

bool TorchHit(uint data, vec3 voxelPos, vec3 worldDir, inout VoxelIntersectOut VIO) {
    Hit2 hit2;
    Ray2 ray2;

    mat4 trans = mat4(1.0);

    ray2.origin = fract(voxelPos);
    ray2.dir = worldDir;

    if (false)
    //if (data != id_torch)
    {
        trans[3].xyz -= 0.5;
        trans = mat4(Mat3XZ(rotate(radians(TorchAngle(data)/4.0*360.0)))) * trans;
        trans[3].xyz += 0.5;

        trans[3].xy += vec2(0.5, -0.5);
        trans = mat4(rotate(radians(30.0))) * trans;

        ray2.origin = (trans * vec4(ray2.origin, 1.0)).xyz;
        ray2.dir = normalize(mat3(trans) * ray2.dir);
    }

    if (data != id_torch) {
        if (data == id_torch_right) {
            ray2.origin.x = 1.0-ray2.origin.x;
            ray2.dir.x = -ray2.dir.x;
        }

        ray2.origin.xy = rotate(radians(30.0)) * (ray2.origin.xy + vec2(0.5, -0.5));
        ray2.dir.xy = rotate(radians(30.0)) * ray2.dir.xy;
    }


    ray2.invDir = 1.0 / ray2.dir;
    vec3 hitPos;
    if (BBoxIntersect(vec2(7 / 16., 0).xyx, vec2(9 / 16., 10 / 16.).xyx, ray2, hit2)) {
        hitPos = ray2.origin + ray2.dir * hit2.tmin;

        // Leave the normal and hitPos in the old space to keep texturing working
        // If we want torches to be properly lit, this function must send out explicit texture space hitpos and normal
        if (false)
        {
            hitPos = (inverse(trans) * vec4(hitPos, 1.0)).xyz;
            hit2.normal = normalize(((inverse(mat3(trans))) * (hit2.normal)).xyz);
        }

        if (all(lessThanEqual(abs(hitPos - vec3(0.5)), vec3(0.5)))) {
            VIO.hit = true;
            VIO.voxelPos = hitPos + floor(voxelPos);

            VIO.plane = hit2.normal;

            return true;
        }
    }

    return false;
}

bool LeavesHit(uint data, vec3 worldDir, vec3 voxelPos2, vec3 plane, vec3 voxelPos3, vec3 plane2, inout VoxelIntersectOut VIO) {
    //*
    vec2 tCoord = ((fract(voxelPos2) * 2.0 - 1.0) * mat2x3(RecoverTangentMat(plane))) * 0.5 + 0.5;
    vec4 diffuse = BlockTexture(tCoord.xy, data, GetFaceIndex(plane), voxelPos2);

    if (diffuse.a < 0.001) {
        tCoord = ((fract(voxelPos3) * 2.0 - 1.0) * mat2x3(RecoverTangentMat(plane2))) * 0.5 + 0.5;
        vec4 diffuse2 = BlockTexture(tCoord.xy, data, GetFaceIndex(plane2), voxelPos3);

        if (diffuse2.a < 0.001) { return false; }
        else {
            VIO.hit = true;
            VIO.voxelPos = voxelPos3;
            VIO.plane = plane2;
            VIO.voxelPos += VIO.plane * 0.02;
            return true;
        }
    }
    else {
        VIO.hit = true;
        VIO.voxelPos = voxelPos2;
        VIO.plane = plane;
        return true;
    }
    //*/
    return false;
}

VoxelIntersectOut VoxelIntersect(vec3 voxelPos, vec3 worldDir) {
    // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf

    int lod = LOCAL_LOD;
    
    if (TRACE_LEAVES && VoxelRead(ivec3(voxelPos), 0) == id_leaves)
    { // Step back by 1 voxel to check current voxel
        vec3 worldDir = -worldDir;
        vec3 bound = ((sign(worldDir) * 0.5 + 0.5) * exp2(lod) - mod(voxelPos, exp2(lod)));
        vec3 temp = bound / worldDir;
        vec3 plane = fMin(temp);
        voxelPos += dot(temp, plane) * worldDir * 1.001;
    }

    ivec3 dir_pos = ivec3(max(sign(worldDir), 0));
    ivec3 uvPos = ivec3(voxelPos);
    ivec3 bound = ((uvPos >> lod) + ivec3(dir_pos)) << lod;

    ivec3 voxel_pos_0 = uvPos;
    vec3 fPos = fract(voxelPos);
    vec3 fPosMAD = fPos / worldDir;

    if (IsPortal(VoxelRead(uvPos, 0))) {
        //if (SubVoxelTrace)
        //SetPhysicalWorldID((g_physicalWorldID + 1) % 2);
        UpdateLogicalWorldID(VoxelRead(uvPos, 0));
        //continue;
    }

    int hit = 0;
    VoxelIntersectOut VIO;
    int steps = 0;

    while (true) {
        vec3 distToBoundary = (bound - voxel_pos_0) * (1.0 / worldDir) - fPosMAD;
        ivec3 uplane = GetMinCompMask(distToBoundary);

        ivec3 isPos = SortMinComp(dir_pos, uplane);

        int nearBound = GetMinComp(bound, uplane);

        ivec3 newPos;
        newPos.z = nearBound + isPos.z - 1;

        float tLength = BinaryDotF(distToBoundary, uplane);
        vec3 temp = fPos + worldDir * tLength;
        vec3 floorTemp = floor(temp);

        if (lod < LOCAL_LOD || OutOfVoxelBounds(newPos.z, uplane) || ++steps > WORLD_SIZE.x) { break; }
        //if (lod < 0 || ++steps > 256) { break; }

        newPos.xy = GetNonMinComps(ivec3(floorTemp) + voxel_pos_0, uplane);
        
        int oldPos = GetMinComp(uvPos, uplane);
        lod += int((newPos.z >> (lod + LOD_STEP)) != (oldPos >> (lod + LOD_STEP))) * LOD_STEP;
        //lod = max(findMSB(newPos.z ^ oldPos), 0);
        lod = min(lod, MAX_LOD); //lod=0;
        uvPos = UnsortMinComp(newPos, uplane);
        //if (any(greaterThan(abs(uvPos.xyz - ivec3(WORLD_SIZE.xyz) / 2), WORLD_SIZE.xyz / 2))) break;

        if (!SUB_VOXEL_TRACE) {
            hit = int(VoxelRead(uvPos, lod) > 0);
        } else {
            uint data = VoxelRead(uvPos, lod);

            if (IsPortal(data)) {
                //if (SubVoxelTrace)
                //SetPhysicalWorldID((g_physicalWorldID + 1) % 2);
                UpdateLogicalWorldID(data);
                continue;
            }

            hit = int(data != 0);
            if (!SubVoxelTrace && IsTorch(data)) hit = 0;
            else if (SubVoxelTrace && hit == 1 && lod == 0) {
                if (IsTorch(data)) {
                    vec3 plane;
                    vec3 voxelPos2 = voxelPos + worldDir * MinComp((bound - voxel_pos_0) * (1.0 / worldDir) - fPosMAD, plane);
                    plane *= sign(-worldDir);
                    voxelPos2 -= plane * 0.001;

                    if (TorchHit(data, voxelPos2, worldDir, VIO)) {
                        return VIO;
                    }
                    else {
                        hit = 0;
                    }
                }
            }

            if (TRACE_LEAVES && data == id_leaves) {
                vec3 plane, plane2;
                vec3 voxelPos2 = voxelPos + worldDir * MinComp((bound - voxel_pos_0) * (1.0 / worldDir) - fPosMAD, plane); plane *= sign(-worldDir); ivec3 stepDir = ivec3(sign(plane));
                vec3 voxelPos3 = voxelPos + worldDir * MinComp((bound - stepDir - voxel_pos_0) * (1.0 / worldDir) - fPosMAD, plane2); plane2 *= sign(-worldDir);

                if (LeavesHit(data, worldDir, voxelPos2, plane, voxelPos3, plane2, VIO)) {
                    return VIO;
                }
                else {
                    hit = 0;
                }
            }
        }



        lod -= hit * LOD_STEP;

        bound.xy = ((newPos.xy >> lod) + isPos.xy) << lod;
        bound.z = nearBound + ((hit - 1) & ((isPos.z * 2 - 1) << lod));
        bound = UnsortMinComp(bound, uplane);
    }

    VIO.hit = bool(hit);
    VIO.voxelPos = voxelPos + worldDir * MinComp((bound - voxel_pos_0) * (1.0 / worldDir) - fPosMAD, VIO.plane);
    VIO.plane *= sign(-worldDir);
    
    return VIO;
}


//#define VoxelIntersect(a, b) VoxelMarchLOD_Oldest(a, b)
VoxelIntersectOut VoxelMarchLOD_Oldest(vec3 rayOrig, vec3 rayDir) {
    //rayOrig += plane * abs(rayOrig) * sign(rayDir) * 0.0001;

    vec3 pos0 = rayOrig;
    vec3 pos = pos0;
    vec3 plane = vec3(0.0);

    vec3 stepDir = sign(rayDir);
    vec3 tDelta = 1.0 / abs(rayDir);

    vec3 tMax0 = ((stepDir * 0.5 + 0.5) - mod(pos0, 1.0)) / rayDir;
    vec3 tMax = tMax0;

    vec3 muls = vec3(0.0);

    float t = 0.0;

    //while (t++ < 128 && all(lessThan(abs(pos.xyz - vec2(128, shadowRadius).yxy), vec2(128, shadowRadius).yxy))) {
    while (t++ < WORLD_SIZE.x && all(lessThan(abs(pos.xyz - WORLD_SIZE.xyz / 2), WORLD_SIZE.xyz / 2))) {
        //if (lookup != 0u) return pos0 + dot(plane, tMax) * rayDir;
        if (VoxelRead(ivec3(pos), 0) > 0) {
        //if (GetVoxel(ivec3(pos), 0) > 0u) {
            VoxelIntersectOut VIO;
            VIO.hit = true;
            VIO.plane = plane;
            muls = muls - plane;
            tMax = tMax0 + tDelta * muls;
            VIO.voxelPos = pos0 + dot(plane, tMax) * rayDir;
            VIO.plane *= sign(-rayDir);
            return VIO;
        }

        plane = fMin(tMax);
        muls = muls + plane;

        tMax = tMax0 + tDelta * muls;
        pos = pos0 + stepDir * muls;
    }

    VoxelIntersectOut VIO;
    VIO.hit = false;

    return VIO;
}

float getPhi(in float y, in float x)
{
    if (x == 0.0) {
        if (y == 0.0) {
            return 0.0;
        }
        else if (y > 0.0) {
            return PI / 2.0;
        }
        else {
            return -1.0 * PI / 2.0;
        }
    }
    else if (x > 0.0) {
        return atan(y / x);
    }
    else if (x < 0.0) {
        if (y >= 0.0) {
            return atan(y / x) + PI;
        }
        else {
            return atan(y / x) + PI;
        }
    }
}

vec3 toPolar(in vec3 cart)
{
    float xySquared = (cart.x * cart.x) + (cart.y * cart.y);
    float radius = sqrt(xySquared + (cart.z * cart.z));
    return vec3(radius, atan(sqrt(xySquared), cart.z), getPhi(cart.y, cart.x));
}

vec3 toCartesian(in vec3 sph) {
    return vec3(
        sin(sph.y) * cos(sph.z) * sph.x,
        sin(sph.y) * sin(sph.z) * sph.x,
        cos(sph.y) * sph.x
    );
}

vec3 ToSphere(vec3 pos) {
    float r = length(pos);
    float theta = atan(pos.y, pos.x);
    float roh = acos(pos.z / r);
    return vec3(r, theta, roh);
}

vec3 ToCart(vec3 a) {
    return a.x * vec3(sin(a.z) * cos(a.y), sin(a.z) * sin(a.y), cos(a.z));
}

vec3 Orb(vec3 pos) {
    const float power = 8.0;
    const float power2 = 8.0;
    const float power3 = 8.0;


    vec3 sphere = (pos);
    sphere.x = pow(sphere.x, power);
    sphere.y *= power2;
    sphere.z *= power3;
    return (sphere);

    float n = power - (float(0.0) / 60.0) * 0;

    float r = length(pos);
    float theta = atan(pos.y, pos.x) * power2;
    float roh = acos(pos.z / r) * (power3 + float(float(0.0) / 360.0) * 2.5 * 0);

    return pow(r, n) * vec3(sin(roh) * cos(theta), sin(roh) * sin(theta), cos(roh));
}

vec3 TheFunction(vec3 pos) {
    //return pos;
    vec3 oldPos = pos;
    vec3 offset = VoxelToWorld(vec3(0)) - cameraPosition * vec3(1,1,1) - trackPos.y * 0;
    pos += offset;

    if (false) {
    pos.xz *= rotate(radians(-30.0));
    pos.xy *= rotate(pos.z / 20.0);
    pos.y -= sin(pos.z / 50.0) * 10.0;
    pos.xy *= rotate(-pos.z / 20.0);
    pos.xz *= rotate(radians(30.0));
    return pos;
    }

    vec3 pos2 = pos;

    if (false)
    {
        float range = 10.0;
        pos.xyz = pos.xyz / WORLD_SIZE.xxx * range;

        vec3 x0 = vec3(0.0, 0.0, 0.0);
        float r = 2.0;

        pos -= x0;
        float theta = atan(pos.z, pos.y);
        pos.yz *= rotate(-theta);
        pos.y -= r;
        float omegle = atan(pos.y, pos.x);
        pos.xy *= rotate(omegle);
        pos.y += r;
        pos.yz *= rotate(theta);
        pos += x0;

        pos.xyz *= WORLD_SIZE.xxx / range;
        pos = mix(pos2, pos, 1.0);
        return pos;
    }

    if (false)
    {
        vec3 oldPos = pos + cameraPosition;
        pos.y -= sin(oldPos.x / 200.0) * sin(oldPos.z / 200.0) * 100.0;
        return pos;
    }
    
    if (false)
    {
        float t4 = pos.y / 2000.0;
        float t5 = -pos.x / 2000.0;
        pos.zy *= mat2(cos(-t4), -sin(-t4), sin(-t4), cos(-t4));
        pos.xz *= mat2(cos(-t5), -sin(-t5), sin(-t5), cos(-t5));
        return pos;
    }

    pos.y += 2.0;


    float K = 3000.0 * (interp(beatFromPos, 265, 271));

    //pos.y += sin(length(pos.xz) / 10.0 - cameraPosition.z / 100.0) * 10.0 * pow(cubesmooth(interp(length(pos.xz), 0.0, 200.0)), 2.0)
    //    * interp(length(pos.xz), K, 0.0);

    float t3 = sin(pos.z * 3.0 / currentSpeed);
    //float t3 = sin(pos.z * 3.0 / 160.0);
    t3 *= distortionIntensity;
    t3 *= (interp(length(pos.xz), K, max(K - 500.0, 0.0)));
    t3 *= mix(1.0, sin(baseFrameCameraPosition.z / 1000.0), interp(beatFromPos, 277, 300));
    pos.xy *= mat2(cos(t3), -sin(t3), sin(t3), cos(t3));
    
    /*
    pos.y += sin(length(pos.z) / 10.0) * 20.0 * cubesmooth(interp(length(pos.x), 0.0, 100.0))
        * pow(interp(beatFromPos, 277-1, 277), 2.0)
        * (pow(interp(beatFromPos, 278.5-1, 278.5), 2.0)*-2.0+1.0)
        * (pow(interp(beatFromPos, 280-1, 280), 2.0)*-2.0+1.0)
        * (pow(interp(beatFromPos, 281.5-1, 281.5), 2.0)*-2.0+1.0)
        * (pow(interp(beatFromPos, 283-1, 283), 2.0)*-2.0+1.0);*/

    return pos;
}

mat3 TheJacobian(vec3 pos) {
    mat3 ret;

    float discrete = 1.0;

    vec3 here = TheFunction(pos);
    ret[0] = (TheFunction(pos + discrete*vec3(1,0,0)) - here) / discrete;
    ret[1] = (TheFunction(pos + discrete*vec3(0,1,0)) - here) / discrete;
    ret[2] = (TheFunction(pos + discrete*vec3(0,0,1)) - here) / discrete;

    return ret;
}

mat3 ArbitraryTBN(vec3 normal) {
    mat3 ret;
    ret[2] = normal;
    ret[0] = normalize(vec3(sqrt(2), sqrt(3), sqrt(5)));
    ret[1] = normalize(cross(ret[0], ret[2]));
    ret[0] = cross(ret[1], ret[2]);

    return ret;
}

vec3 CalculateConeVector(const float i, const float angularRadius, const int steps) {
    float x = i * 2.0 - 1.0;
    float y = i * float(steps) * 1.618 * 256.0;

    float angle = acos(x) * angularRadius / 3.14159;
    float s = sin(angle);

    return vec3(cos(y) * s, sin(y) * s, cos(angle));
}

bool IsAirBlock(uint data) {
    return data > 0 && !IsPortal(data);
}

bool VoxelBoolRead2(ivec3 pos, int lod) {
    if (SAMPLE_COUNT == 1 || (sampledFrameID % SAMPLE_COUNT) != 0) {
        return false;
    }

    int oldWorldID = g_physicalWorldID;

    bool result = false;
    for (int i = 0; i < 4; ++i) {
        uint data = VoxelRead(ivec3(pos) + ivec3(0, 0, i), lod);
        if (false&&IsPortal(data)) {
            //SetPhysicalWorldID((g_physicalWorldID + 1) % 2);
            
        } else if (data > 0) {
            result = true;
            break;
        }
    }

    //SetPhysicalWorldID(oldWorldID);

    return result;
}

bool Offscreen(ivec2 offset) {
    ivec2 coord = ivec2(gl_FragCoord.xy) + offset;
    return any(greaterThan(coord, ivec2(viewSize.xy)-1))
        || any(lessThan(coord, ivec2(0)));
}

vec3 g_voxelPos;
VoxelIntersectOut VoxelMarchLOD(vec3 rayOrig, inout vec3 rayDir, float renderDistance) {
    if (distortionIntensity <= 0.0) {
        return VoxelIntersect(rayOrig, rayDir);
    }

    bool saveData = false;

    vec3 pos = rayOrig;
    vec3 pos4 = pos*0;
    vec3 pos0 = rayOrig;
    vec3 plane = vec3(0.0);

    float t = 0.0;

    vec3 rayDir2 = rayDir;
    float tAccum = 0.0;
    
    const int lod = 0;

    //*
    if (DistortionReuse()) {
    saveData = true;
    if ((sampledFrameID % SAMPLE_COUNT)!= 0) {
    {
        vec4 curr0 = imageLoad(distortionReuseImage, ivec2(gl_FragCoord.xy));
        vec4 curr1 = imageLoad(distortionReuseImage, ivec2(gl_FragCoord.xy) + ivec2( 0, 1)) + float(Offscreen(ivec2( 0, 1)))*vec4(0,0,0,1e10);
        vec4 curr2 = imageLoad(distortionReuseImage, ivec2(gl_FragCoord.xy) + ivec2( 0,-1)) + float(Offscreen(ivec2( 0,-1)))*vec4(0,0,0,1e10);
        vec4 curr3 = imageLoad(distortionReuseImage, ivec2(gl_FragCoord.xy) + ivec2( 1, 0)) + float(Offscreen(ivec2( 1, 0)))*vec4(0,0,0,1e10);
        vec4 curr4 = imageLoad(distortionReuseImage, ivec2(gl_FragCoord.xy) + ivec2(-1, 0)) + float(Offscreen(ivec2(-1, 0)))*vec4(0,0,0,1e10);

        float nearest = min(min(min(min(curr0.w, curr1.w), curr2.w), curr3.w), curr4.w);

        pos = curr0.xyz; tAccum = curr0.w;

        bool edgeDetect = true;

        vec2 delta = vec2(0.0);

        if (edgeDetect) {
        if (nearest == curr1.w) { pos = curr1.xyz; tAccum = curr1.w; delta = vec2( 0, 1); }
        if (nearest == curr2.w) { pos = curr2.xyz; tAccum = curr2.w; delta = vec2( 0,-1); }
        if (nearest == curr3.w) { pos = curr3.xyz; tAccum = curr3.w; delta = vec2( 1, 0); }
        if (nearest == curr4.w) { pos = curr4.xyz; tAccum = curr4.w; delta = vec2(-1, 0); }
        }

        if (tAccum >= 1e10) {
            VoxelIntersectOut VIO;
            VIO.hit = false;
            return VIO;
        }
        
        vec3 pos5 = pos;
        vec3 pos6 = VoxelToWorld(rayOrig + rayDir * tAccum) - cameraPosition;

        if (true||tAccum > 10.0)
        {
            pos = pos6;
            vec2 hash = TAAHash(sampledFrameID % 16384) * float((sampledFrameID % SAMPLE_COUNT) != 0);
            
            pos = Project(pos);
            
            pos.xy /= pos.z;
            
            pos.xy = gl_FragCoord.xy / viewSize.xy * 2.0 - 1.0;
            pos.xy -= delta / viewSize.xy * 2.0;
            
            pos.xy += hash * 1.0;
            pos.xy *= pos.z;
            pos = Unproject(pos);

            vec3 delta = pos - pos6;

            delta = TheJacobian(rayOrig + rayDir * tAccum) * delta;

            pos = WorldToVoxel(pos5.xyz + delta + cameraPosition);

            for (int i = 0; i < 1; ++i) {
                float discrete = 1;

                int maxSamples = int(exp2(lod)) * 1;
                for (int i = 0; i < maxSamples; ++i) {
                    rayDir2 = ((TheFunction(rayOrig + rayDir * (tAccum + discrete)) - TheFunction(rayOrig + rayDir * tAccum))) / discrete;
                    rayDir2 = mix(rayDir2, rayDir, 0.00001); // Fixes weird seam line when derivative is near zero
                    //rayDir2 = rayDir;
                    tAccum -= 1.0;
                    pos -= rayDir2;
                }
            }
    
            {
                float discrete = 1;

                int maxSamples = int(exp2(lod)) * 1;
                for (int i = 0; i < maxSamples; ++i) {
                    rayDir2 = ((TheFunction(rayOrig + rayDir * (tAccum + discrete)) - TheFunction(rayOrig + rayDir * tAccum))) / discrete;
                    rayDir2 = mix(rayDir2, rayDir, 0.00001); // Fixes weird seam line when derivative is near zero
                    vec3 bound = ((sign(rayDir2) * 0.5 + 0.5) * exp2(lod) - mod(pos, exp2(lod)));
                    vec3 temp = bound / rayDir2 * (1.0 / (maxSamples - i));
                    plane = fMin(temp);
                    tAccum += dot(temp, plane);
                    pos += dot(temp, plane) * rayDir2;// +plane * sign(rayDir2) * 0.0001 * float(WORLD_SIZE.z / 2048.0) * (i == maxSamples - 1 ? 1.0 : 0.0);
                }
            }
        }
        else {
            pos = rayOrig;
            tAccum = 0.0;
        }
    }


    saveData = false;
    }
    }
    //*/
    
    while (t++ < int(2 * WORLD_SIZE.x * renderDistance)
        && ( BOUNDS_CHECKING || all(lessThan(abs(pos.xyz - WORLD_SIZE.xyz / 2), WORLD_SIZE.xyz / 2)) )
        ) {

        if (!BOUNDS_CHECKING || all(lessThan(abs(pos.xyz - WORLD_SIZE.xyz / 2), WORLD_SIZE.xyz / 2))) {


            bool hit;
            
            hit = VoxelBoolRead2(ivec3(pos), lod);

            if (!SUB_VOXEL_TRACE) {
                
            }
            else if (!(SAMPLE_COUNT > 1 && ((sampledFrameID % SAMPLE_COUNT) == 0))) {
                
                uint data = VoxelRead(ivec3(pos), lod);

                if (IsPortal(data)) {
                    UpdateLogicalWorldID(data);
                    continue;
                }

                hit = data != 0;
                if (!SubVoxelTrace && IsTorch(data)) hit = false;
                else if (SubVoxelTrace && hit && lod == 0) {
                    if (IsTorch(data)) {
                        vec3 plane2 = plane;
                        vec3 voxelPos2 = pos + plane * sign(rayDir2) * 0.002;
                        
                        VoxelIntersectOut VIO;
                        if (TorchHit(data, voxelPos2, rayDir, VIO)) {
                            return VIO;
                        }
                        else {
                            hit = false;
                        }
                    }
                }

                //*
                if (TRACE_LEAVES && data == id_leaves) {
                    vec3 plane2;
                    vec3 voxelPos2 = pos + plane * sign(rayDir2) * 0.002;
                    //vec3 voxelPos2 = voxelPos + worldDir * MinComp((bound - voxel_pos_0) * (1.0 / worldDir) - fPosMAD, plane); plane *= sign(-worldDir); ivec3 stepDir = ivec3(sign(plane));
                    //vec3 voxelPos3 = voxelPos + worldDir * MinComp((bound - stepDir - voxel_pos_0) * (1.0 / worldDir) - fPosMAD, plane2); plane2 *= sign(-worldDir);

                    VoxelIntersectOut VIO;
                    if (LeavesHit(data, rayDir2, voxelPos2, plane*sign(rayDir2), vec3(0), vec3(0), VIO)) {
                        return VIO;
                    }
                    else {
                        hit = false;
                    }
                }
                //*/
            }
            
            

            if (hit) {
                VoxelIntersectOut VIO;
                VIO.hit = true;
                VIO.voxelPos = pos - plane * sign(rayDir2) * 0.002;
                g_voxelPos = rayOrig + rayDir *tAccum;
                rayDir = rayDir2;
                VIO.plane = plane * sign(-rayDir);
                if (saveData) {
                    vec4 curr;
                    curr.xyz = VoxelToWorld(pos) - cameraPosition;
                    curr.w = tAccum;
                    
                    imageStore(distortionReuseImage, ivec2(gl_FragCoord.xy), curr);
                }
                return VIO;
            }
        }

        float discrete = 1;

        int maxSamples = int(exp2(lod)) * CURVATURE_SAMPLES;
        for (int i = 0; i < maxSamples; ++i) {
            rayDir2 = ((TheFunction(rayOrig + rayDir * (tAccum + discrete)) - TheFunction(rayOrig + rayDir * tAccum))) / discrete;
            rayDir2 = mix(rayDir2, rayDir, 0.00001); // Fixes weird seam line when derivative is near zero
            vec3 bound = ((sign(rayDir2) * 0.5 + 0.5) * exp2(lod) - mod(pos, exp2(lod)));
            vec3 temp = bound / rayDir2 * (1.0 / (maxSamples - i));
            plane = fMin(temp);
            tAccum += dot(temp, plane);
            pos += dot(temp, plane) * rayDir2;
            pos4 += dot(temp, plane) * abs(rayDir2);
        }
        pos += plane * sign(rayDir2) * float(WORLD_SIZE.z * 2e-7);
    }

    if (saveData)
    {
        vec4 curr;
        curr.xyz = VoxelToWorld(pos) - cameraPosition;
        curr.w = 1e10;
        imageStore(distortionReuseImage, ivec2(gl_FragCoord.xy), curr);
    }

    VoxelIntersectOut VIO;
    VIO.hit = false;

    return VIO;
}

struct RayStruct {
    vec3 voxelPos;
    vec3 worldDir;
    vec3 absorb;
    uint info;
    int world_ID;
};

#define DIFFUSE_ACCUM_INDEX 0
#define SPECULAR_ACCUM_INDEX 1

const uint    PRIMARY_RAY_TYPE = (1 << 8);
const uint   SUNLIGHT_RAY_TYPE = (1 << 9);
const uint    AMBIENT_RAY_TYPE = (1 << 10);
const uint   SPECULAR_RAY_TYPE = (1 << 11);
const uint UNDERWATER_RAY_TYPE = (1 << 12);

const uint RAY_DEPTH_MASK = (1 << 8) - 1;
const uint RAY_TYPE_MASK = ((1 << 16) - 1) & (~RAY_DEPTH_MASK);
const uint RAY_ATTR_MASK = ((1 << 24) - 1) & (~RAY_DEPTH_MASK) & (~RAY_TYPE_MASK);

const uint SPECULAR_RAY_ATTR = (1 << 16);

#define MAX_RAYS 4
#define MAX_RAY_BOUNCES 1
#define SAMPLES 1
#define RAY_STACK_CAPACITY (MAX_RAY_BOUNCES*2 + 2)
RayStruct rayQueue[RAY_STACK_CAPACITY];
int  rayStackTop = 0;
bool stackOutOfSpace = false;
bool IsStackFull() { return rayStackTop == RAY_STACK_CAPACITY; }
bool IsStackEmpty() { return rayStackTop == 0; }

uint PackRayInfo(uint rayDepth, const uint RAY_TYPE) {
    return rayDepth | RAY_TYPE;
}

uint PackRayInfo(uint rayDepth, const uint RAY_TYPE, uint RAY_ATTR) {
    return rayDepth | RAY_TYPE | RAY_ATTR;
}

uint GetRayType(uint info) {
    return info & RAY_TYPE_MASK;
}

bool IsAmbientRay(RayStruct ray) { return ((ray.info & AMBIENT_RAY_TYPE) != 0); }
bool IsSunlightRay(RayStruct ray) { return ((ray.info & SUNLIGHT_RAY_TYPE) != 0); }
bool IsPrimaryRay(RayStruct ray) { return ((ray.info & PRIMARY_RAY_TYPE) != 0); }
bool IsSpecularRay(RayStruct ray) { return ((ray.info & SPECULAR_RAY_TYPE) != 0); }
bool IsUnderwaterRay(RayStruct ray) { return ((ray.info & UNDERWATER_RAY_TYPE) != 0); }

uint GetRayAttr(uint info) {
    return info & RAY_ATTR_MASK;
}

bool HasRayAttr(uint info, const uint RAY_ATTR) {
    return (info & RAY_ATTR) != 0;
}

uint GetRayDepth(uint info) {
    return info & RAY_DEPTH_MASK;
}

void RayPush(RayStruct elem) {
    stackOutOfSpace = stackOutOfSpace || IsStackFull();
    if (stackOutOfSpace) return;
    //if (!PassesVisibilityThreshold(elem.absorb)) { return; }

    rayQueue[rayStackTop % RAY_STACK_CAPACITY] = elem;
    ++rayStackTop;
    return;
}

RayStruct RayPop() {
    --rayStackTop;
    RayStruct res = rayQueue[rayStackTop % RAY_STACK_CAPACITY];
    return res;
}

struct ColorStruct {
    vec3 color;
    vec3 sky;
    float fogfactor;
};

struct ColorStruct2 {
    float fogfactor;
};

ColorStruct primary = ColorStruct(vec3(0), vec3(0), 1.0);
ColorStruct2 specular = ColorStruct2(1.0);

bool BlockOccupied(ivec3 voxelPos) {
    uint data = VoxelRead(voxelPos, 0);
    return data != 0 && !IsTorch(data);
}

float Chroma() {
    return float((sampledFrameID % 16384) % 3) / 3.0;
}

bool Screenshot() {
    return false;
    return (sampledFrameID % sampleCount) != 0;
}

void NewFunction(vec2 uv2) {
    vec2 hash = TAAHash(sampledFrameID % 16384)
        * float((sampledFrameID % sampleCount) != 0)            // Always disable hash for sample 0
        * float(!DistortionReuse())                             // Disable hash for distortion re-use. It will be applied later.
        ;

    vec3 sunColor = vec3(1.5, 1.0, 1.0) * 2.0;

    vec3 normal;

    vec3 worldDir; {
        worldDir = vec3(uv2 * 2.0 - 1.0, 1.0);
        
        if (Screenshot()) {
            worldDir.xy *= 0.95 + Chroma() * max(1.0, (distance(worldDir.xy, Fisheye(worldDir.xy)))) / 100.0;
            hash *= max(1.0, 4 * (distance(worldDir.xy, Fisheye(worldDir.xy))));
        }
        worldDir.xy += hash;
        worldDir.xy = (Fisheye(worldDir.xy));

        worldDir = normalize(Unproject(worldDir));
    }

    mat3 tanMat;

    RayStruct curr;

    curr.world_ID = g_logicalWorldID;
    curr.absorb = vec3(1.0);
    curr.worldDir = worldDir;
    curr.voxelPos = cameraPosition;
    if (curr.voxelPos.y > WORLD_SIZE.y) {
        curr.voxelPos = curr.voxelPos + curr.worldDir * (WORLD_SIZE.y - 0.001 - curr.voxelPos.y) / curr.worldDir.y;
    }
    curr.voxelPos = WorldToVoxel(curr.voxelPos);
    VoxelIntersectOut VIO = VoxelMarchLOD(curr.voxelPos, curr.worldDir, 1.0);
    curr.world_ID = g_logicalWorldID;
    curr.worldDir = normalize(curr.worldDir);

    vec3 primaryWorldPos = VoxelToWorld(VIO.voxelPos) - cameraPosition;

    primary.fogfactor = VIO.hit ? FogFactor(primaryWorldPos) : 1.0;
    
    vec3 sunspot;
    vec3 clouds;
    primary.sky = ComputeTotalSky(vec3(0.0), worldDir, sunspot, clouds) / SKY_MULT;
    
    OutColor.rgb += clouds * pow(primary.fogfactor, 8.0);
    OutColor1.rgb += sunspot * pow(primary.fogfactor, 16.0);

    if (!VIO.hit) { return; }
    
    SubVoxelTrace = false;

    vec3 wPosVector = VoxelToWorld(g_voxelPos) - cameraPosition;
    
    tanMat = RecoverTangentMat(VIO.plane);
    vec2 tCoord;

    uint data = VoxelRead(ivec3(VIO.voxelPos - VIO.plane * 0.01), 0);
    //if (data == id_leaves) VIO.voxelPos += sunDirection * 0.01;
    curr.voxelPos = VIO.voxelPos + VIO.plane * 0.001;
    if (data == id_water) {
        
        if (distortionIntensity <= 0.0) {
            wPosVector = VoxelToWorld(VIO.voxelPos) - cameraPosition;
        }

        if (distortionIntensity > 0.0 || true) {
            tanMat[0] = TheFunction(curr.voxelPos + tanMat[0]) - TheFunction(curr.voxelPos);
            tanMat[1] = TheFunction(curr.voxelPos + tanMat[1]) - TheFunction(curr.voxelPos);
            tanMat[2] = normalize(-cross(tanMat[0], tanMat[1])) * vec3(-1, 1, -1);
            tanMat[1] = normalize(cross(tanMat[0], tanMat[2]));
            tanMat[0] = normalize(tanMat[0]);
            normal = tanMat[2];
        } else {
            wPosVector = primaryWorldPos;
            normal = vec3(0, 1, 0);
        }

        normal = tanMat * ComputeWaveNormals(primaryWorldPos, normal);
        
        curr.worldDir = reflect(normalize(wPosVector), normal);
        curr.info = PackRayInfo(0, PRIMARY_RAY_TYPE, SPECULAR_RAY_ATTR);
        
        {
            vec3 oldWorldDir = normalize(curr.worldDir);
            //SetPhysicalWorldID(curr.world_ID);
            SetLogicalWorldID(curr.world_ID);
            //VoxelIntersectOut VIO = VoxelMarchLOD(curr.voxelPos, curr.worldDir, 1.0);
            VoxelIntersectOut VIO = VoxelIntersect(curr.voxelPos, curr.worldDir);
            curr.world_ID = g_logicalWorldID;
            curr.worldDir = normalize(curr.worldDir);
            
            data = VoxelRead(ivec3(VIO.voxelPos - VIO.plane * 0.01), 0);
            if (!VIO.hit || data == id_water) {} else specular.fogfactor = FogFactor(VoxelToWorld(VIO.voxelPos) - cameraPosition);

            float ior = 1.0 - pow(max(0.0, dot(normalize(-wPosVector), normal)), 1.0);

            vec3 specSunspot;
            vec3 specClouds;
            vec3 specSky = ior * ComputeTotalSky(wPosVector, oldWorldDir, specSunspot, specClouds) / SKY_MULT;
            specSunspot *= ior;
            specClouds *= ior;

            if (DO_ATMOSPHERE) OutColor.rgb += vec3(0.2, 0.2, 1.0)*0.3 * (1.0 - exp(-length(VoxelToWorld(VIO.voxelPos) - cameraPosition)/WORLD_SIZE.x)) * (1.0 - primary.fogfactor) ;
            OutColor.rgb += specSky * (1.0 - primary.fogfactor) * specular.fogfactor;
            OutColor.rgb += specClouds * pow(specular.fogfactor, 8.0) * (1.0 - primary.fogfactor);

            OutColor1.rgb += specSunspot * pow(specular.fogfactor, 16.0) * (1 - primary.fogfactor);
            
            if (!VIO.hit || data == id_water) { return; }
            VIO.plane *= VIO.plane * sign(-curr.worldDir);
            
            
            tanMat = RecoverTangentMat(VIO.plane);
            curr.voxelPos = VIO.voxelPos + VIO.plane * 0.001;

            tCoord = ((fract(VIO.voxelPos) * 2.0 - 1.0) * mat2x3(tanMat)) * 0.5 + 0.5;
            vec4 diffuse = BlockTexture(tCoord.xy, data, GetFaceIndex(VIO.plane), VIO.voxelPos);

            curr.absorb *= diffuse.rgb;
            curr.absorb *= ior * (1.0 - specular.fogfactor);

            
            normal = VIO.plane;
        }

    } else {
        if (DO_ATMOSPHERE) OutColor.rgb += vec3(0.2, 0.2, 1.0) * 0.3 * (1.0 - exp(-length(primaryWorldPos) / WORLD_SIZE.x)) * (1.0 - primary.fogfactor);
        normal = VIO.plane;

        if (data == id_beat) {
            primary.color += 1.0;
        }

        tanMat = RecoverTangentMat(VIO.plane);
        tCoord = ((fract(VIO.voxelPos) * 2.0 - 1.0) * mat2x3(tanMat)) * 0.5 + 0.5;
        vec4 diffuse = BlockTexture(tCoord.xy, data, GetFaceIndex(VIO.plane), VIO.voxelPos);
        curr.absorb *= diffuse.rgb;
        curr.info = PackRayInfo(0, PRIMARY_RAY_TYPE);
    }

    //if (false) // Torch lighting
    {
        vec3 pos = VoxelToWorld(VIO.voxelPos - VIO.plane * 0.001);


        float torchPos = int(beatsSSBO[BinarySearchNearest(int(pos.z))].zPos);
        float dist = distance(vec3(trackPos.xy+vec2(0,2),torchPos), pos);
        float torchDist = pow((asin(sin(pos.z / 10.0)) / 3.14159 * 2.0 * 0.5 + 0.5), 8.0);
        torchDist = pow(clamp01(1.0 - dist /16.0), 4.0);
        float torchBrightness = 4.0 * torchDist;
        if (torchBrightness > 0.01) {
            float r = float(BlockOccupied(ivec3(curr.voxelPos) +ivec3(tanMat[0]))) * (tCoord.x);
            float l = float(BlockOccupied(ivec3(curr.voxelPos) -ivec3(tanMat[0]))) * (1 - tCoord.x);
            float t = float(BlockOccupied(ivec3(curr.voxelPos) +ivec3(tanMat[1]))) * (tCoord.y);
            float b = float(BlockOccupied(ivec3(curr.voxelPos) -ivec3(tanMat[1]))) * (1 - tCoord.y);

            float tr = float(BlockOccupied(ivec3(curr.voxelPos) +ivec3(tanMat[1])+ivec3(tanMat[0])));
            float tl = float(BlockOccupied(ivec3(curr.voxelPos) +ivec3(tanMat[1])-ivec3(tanMat[0])));
            float br = float(BlockOccupied(ivec3(curr.voxelPos) -ivec3(tanMat[1])+ivec3(tanMat[0])));
            float bl = float(BlockOccupied(ivec3(curr.voxelPos) -ivec3(tanMat[1])-ivec3(tanMat[0])));

            float c_tr = t / 3.0 + r / 3.0 + (tr) / 3.0;
            float c_tl = t / 3.0 + l / 3.0 + (tl) / 3.0;
            float c_br = b / 3.0 + r / 3.0 + (br) / 3.0;
            float c_bl = b / 3.0 + l / 3.0 + (bl) / 3.0;

            float amt_tr = max(0.0, +tCoord.x + tCoord.y - 1.0);
            float amt_tl = max(0.0, -tCoord.x + tCoord.y - 1.0 + 1.0);
            float amt_br = max(0.0, +tCoord.x - tCoord.y - 1.0 + 1.0);
            float amt_bl = max(0.0, -tCoord.x - tCoord.y - 1.0 + 2.0);

    #define idunno(a, b) (1.0-((1.0-(a)) * (1.0 - (b))))
    //#define idunno(a, b) ((a) + (b))
            
            float ret = 0.0;
            ret = max(ret, max(idunno(r, b), amt_br * br));
            ret = max(ret, max(idunno(r, t), amt_tr * tr));
            ret = max(ret, max(idunno(l, b), amt_bl * bl));
            ret = max(ret, max(idunno(l, t), amt_tl * tl));
            ret = 1.0 - ret * 0.7;
            if (IsTorch(data)) ret = 1.0;
        
            primary.color += curr.absorb * vec3(1.0, 0.5, 0.0) * torchBrightness * ret;
        }
    }

    if (FAST_LIGHTING) {
        float right = (float(VoxelRead(ivec3(curr.voxelPos) + ivec3(tanMat[0]), 0) > 0) * (tCoord.x));
        float left = (float(VoxelRead(ivec3(curr.voxelPos) + ivec3(-tanMat[0]), 0) > 0) * (1 - tCoord.x));
        float top = (float(VoxelRead(ivec3(curr.voxelPos) + ivec3(tanMat[1]), 0) > 0) * (tCoord.y));
        float bottom = (float(VoxelRead(ivec3(curr.voxelPos) + ivec3(-tanMat[1]), 0) > 0) * (1 - tCoord.y));

        float light = idunno(left, bottom) * idunno(top, right) * idunno(right, bottom) * idunno(left, top);
        light = mix(light, 1.0, 0.4);
        
        curr.absorb *= light;
        
        RayStruct ambRay = curr;
        ambRay.worldDir = vec3(0.0, 1.0, 0.0);
        ambRay.info = PackRayInfo(GetRayDepth(curr.info) + 1, AMBIENT_RAY_TYPE, GetRayAttr(curr.info));
        
        
        primary.color += ComputeTotalSky2(ambRay.worldDir) * ambRay.absorb;

        RayStruct sunRay = curr;
        sunRay.worldDir = sunDirection;
        sunRay.absorb *= sunIrradiance * max(0.0, dot(sunRay.worldDir, normal));
        sunRay.info = PackRayInfo(GetRayDepth(curr.info) + 1, SUNLIGHT_RAY_TYPE, GetRayAttr(curr.info));
        //SetPhysicalWorldID(sunRay.world_ID);
        SetLogicalWorldID(sunRay.world_ID);
        VoxelIntersectOut VIO = VoxelIntersect(sunRay.voxelPos, sunRay.worldDir);
        sunRay.world_ID = g_logicalWorldID;

        if (!VIO.hit) {
            vec3 c = sunRay.absorb * 4.0 / SAMPLES;
            if (GetRayAttr(sunRay.info) == SPECULAR_RAY_ATTR) OutColor.rgb += c;
            else primary.color += c;
        }

        return;
    }

    RayStruct sunRay = curr;
    //sunRay.worldDir = (ArbitraryTBN(sunDirection) * CalculateConeVector(RandNextF(), radians(0.0), 32));
    sunRay.worldDir = sunDirection;
    if (data != id_leaves) sunRay.absorb *= max(0.0, dot(sunRay.worldDir, normal));
    sunRay.absorb *= sunIrradiance;
    sunRay.info = PackRayInfo(GetRayDepth(curr.info) + 1, SUNLIGHT_RAY_TYPE, GetRayAttr(curr.info));
    if (length(sunRay.absorb) > 0.001)
        RayPush(sunRay);

    for (int sam = 0; sam < SAMPLES; ++sam)
    {
        RayStruct ambRay = curr;
        ambRay.worldDir = ArbitraryTBN(normal) * CalculateConeVector(RandNextF(), radians(90.0), 32);
        ambRay.info = PackRayInfo(GetRayDepth(curr.info) + 1, AMBIENT_RAY_TYPE, GetRayAttr(curr.info));
        RayPush(ambRay);

        int RAY_COUNT;
        for (RAY_COUNT = 1; RAY_COUNT < MAX_RAYS; ++RAY_COUNT) {
            if (IsStackEmpty()) { RAY_COUNT = MAX_RAYS; continue; }
            RayStruct thisRay = RayPop();
            vec3 oldWorldDir = thisRay.worldDir;
            //SetPhysicalWorldID(thisRay.world_ID);
            SetLogicalWorldID(sunRay.world_ID);
            VoxelIntersectOut VIO = VoxelIntersect(thisRay.voxelPos, thisRay.worldDir);
            thisRay.world_ID = g_logicalWorldID;
            bool isPrimary = GetRayDepth(thisRay.info) > 1;

            if (IsSunlightRay(thisRay)) {
                vec3 c = thisRay.absorb * (isPrimary ? 4.0 : 1.0) / SAMPLES;
                if (VIO.hit) c *= pow(FogFactor(VoxelToWorld(VIO.voxelPos) - cameraPosition), 16.0);
                if (GetRayAttr(thisRay.info) == SPECULAR_RAY_ATTR) OutColor.rgb += c;
                else primary.color += c;
                continue;
            }

            if (!VIO.hit) {
                vec3 c = ComputeTotalSky2(oldWorldDir) * thisRay.absorb / SAMPLES / (isPrimary ? SKY_MULT : 1.0);
                if (GetRayAttr(thisRay.info) == SPECULAR_RAY_ATTR) OutColor.rgb += c;
                else primary.color += c;
                continue;
            }

            VIO.plane *= VIO.plane * sign(-thisRay.worldDir);
            mat3 tanMat = RecoverTangentMat(VIO.plane);
            vec2 tCoord = ((fract(VIO.voxelPos) * 2.0 - 1.0) * mat2x3(RecoverTangentMat(VIO.plane))) * 0.5 + 0.5;
            uint data = VoxelRead(ivec3(VIO.voxelPos - VIO.plane * 0.01), 0);
            vec4 diffuse = BlockTexture(tCoord.xy, data, GetFaceIndex(VIO.plane), VIO.voxelPos);

            if (data == id_beat) {
                primary.color += 1.0*thisRay.absorb;
            }

            thisRay.absorb *= diffuse.rgb;
            thisRay.voxelPos = VIO.voxelPos + VIO.plane * 0.001;

            RayStruct sunRay = thisRay;
            sunRay.worldDir = sunDirection;
            if (data != id_leaves) sunRay.absorb *= max(0.0, dot(sunRay.worldDir, VIO.plane));
            sunRay.absorb *= sunIrradiance;
            sunRay.info = PackRayInfo(GetRayDepth(thisRay.info) + 1, SUNLIGHT_RAY_TYPE, GetRayAttr(thisRay.info));
            
            if (length(sunRay.absorb) > 0.001) RayPush(sunRay);
            if (GetRayDepth(thisRay.info) >= MAX_RAY_BOUNCES) continue;

            RayStruct ambRay = thisRay;
            ambRay.worldDir = ArbitraryTBN(VIO.plane) * CalculateConeVector(RandNextF(), radians(90.0), 32);
            ambRay.info = PackRayInfo(GetRayDepth(thisRay.info) + 1, AMBIENT_RAY_TYPE, GetRayAttr(thisRay.info));
            RayPush(ambRay);
        }
    }
}

void main() {
    //SetPhysicalWorldID(uWorldID);
    SetLogicalWorldID(uWorldID);
    randState = triple32(uint(gl_FragCoord.x) * 12345 + uint(gl_FragCoord.y) + floatBitsToUint(sampledFrameID % 16384) * 123456789);
    //if ((sampledFrameID % 8) == 7) return;
    OutColor.rgba = vec4(0.0, 0.0, 0.0, 1.0);
    OutColor1.rgba = vec4(0.0, 0.0, 0.0, 1.0);

    NewFunction(uv);

    if (SAMPLE_COUNT > 1 && (sampledFrameID % SAMPLE_COUNT) == 0
        && distortionIntensity > 0.0) {
        return;
    }
    
    OutColor.rgb += 0.0
        + primary.color  * (1.0 - primary.fogfactor)      //* specular.fogfactor
        + primary.sky    *        primary.fogfactor       * specular.fogfactor
    //    + primary.clouds * pow(   primary.fogfactor, 8.0) //* specular.fogfactor
    //    + specular.color * (1.0 - specular.fogfactor)
    //    + specular.sky * (1.0 - primary.fogfactor) * specular.fogfactor
    //    + specular.clouds * pow(specular.fogfactor, 8.0) * (1.0 - primary.fogfactor)
        ;

    OutColor1.rgb += 0.0
    //    + specular.sunspot * pow(specular.fogfactor, 16.0) * (1 - primary.fogfactor)
    //    + primary.sunspot * pow(primary.fogfactor, 16.0)
        ;

    OutColor.rgb = max(vec3(0.0), OutColor.rgb);
    OutColor1.rgb = max(vec3(0.0), OutColor1.rgb);
    
    if (Screenshot()) {
        OutColor.rgb = rgb(hsv(OutColor.rgb));
        vec3 col = hsv(((sampledFrameID % 16384) % 3) == 0 ? vec3(1, 0, 0) : ((sampledFrameID % 16384) % 3) == 1 ? vec3(0, 1, 0) : vec3(0, 0, 1));
        col = rgb(col+vec3(4/6.0,0,0)) * 3.0 * 1.0;

        OutColor.rgb *= mix(vec3(1.0), col, 0.7);
        OutColor1.rgb *= mix(vec3(1.0), col, 0.7);
    }

    exit();

    imageStore(OutColor0Image, ivec2(gl_GlobalInvocationID.xy), imageLoad(OutColor0Image, ivec2(gl_GlobalInvocationID.xy)) + OutColor);
    imageStore(OutColor1Image, ivec2(gl_GlobalInvocationID.xy), imageLoad(OutColor1Image, ivec2(gl_GlobalInvocationID.xy)) + OutColor1);
}

#endif
#endif
