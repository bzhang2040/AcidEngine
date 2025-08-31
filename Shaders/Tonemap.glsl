#ifdef CXX_STAGE
    #define Tonemap_glsl "Tonemap.glsl", "TONEMAP_STAGE", "graphics"
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


/***********************************************************************/
/* Text Rendering */
const int
	_A    = 0x64bd29, _B    = 0x749d27, _C    = 0xe0842e, _D    = 0x74a527,
	_E    = 0xf09c2f, _F    = 0xf09c21, _G    = 0xe0b526, _H    = 0x94bd29,
	_I    = 0xf2108f, _J    = 0x842526, _K    = 0x928CA9, _L    = 0x10842f,
	_M    = 0x97a529, _N    = 0x95b529, _O    = 0x64a526, _P    = 0x74a4e1,
	_Q    = 0x64acaa, _R    = 0x749ca9, _S    = 0xe09907, _T    = 0xf21084,
	_U    = 0x94a526, _V    = 0x94a544, _W    = 0x94a5e9, _X    = 0x949929,
	_Y    = 0x94b90e, _Z    = 0xf4106f, _0    = 0x65b526, _1    = 0x431084,
	_2    = 0x64904f, _3    = 0x649126, _4    = 0x94bd08, _5    = 0xf09907,
	_6    = 0x609d26, _7    = 0xf41041, _8    = 0x649926, _9    = 0x64b904,
	_APST = 0x631000, _PI   = 0x07a949, _UNDS = 0x00000f, _HYPH = 0x001800,
	_TILD = 0x051400, _PLUS = 0x011c40, _EQUL = 0x0781e0, _SLSH = 0x041041,
	_EXCL = 0x318c03, _QUES = 0x649004, _COMM = 0x000062, _FSTP = 0x000002,
	_QUOT = 0x528000, _BLNK = 0x000000, _COLN = 0x000802, _LPAR = 0x410844,
	_RPAR = 0x221082;

const ivec2 MAP_SIZE = ivec2(5, 5);

const float rLog10 = 1.0 / log(10.0);
#define log10(x) (log(x) * rLog10)

int GetBit(int bitMap, int index) {
	return (bitMap >> index) & 1;
}

float DrawChar(int charBitMap, inout vec2 anchor, vec2 charSize, vec2 uv) {
	uv = (uv - anchor) / charSize;
	
	anchor.x += charSize.x;
	
	if (!all(lessThan(abs(uv - vec2(0.5)), vec2(0.5))))
		return 0.0;
	
	uv *= MAP_SIZE;
	
	int index = int(uv.x) % MAP_SIZE.x + int(uv.y)*MAP_SIZE.x;
	
	return GetBit(charBitMap, index);
}

const int STRING_LENGTH = 8;
int string_to_draw[STRING_LENGTH];

float DrawString(inout vec2 anchor, vec2 charSize, int stringLength, vec2 uv) {
	uv = (uv - anchor) / charSize;
	
	anchor.x += charSize.x * stringLength;
	
	if (!all(lessThan(abs(uv / vec2(stringLength, 1.0) - vec2(0.5)), vec2(0.5))))
		return 0.0;
	
	int charBitMap = string_to_draw[int(uv.x)];
	
	uv *= MAP_SIZE;
	
	int index = int(uv.x) % MAP_SIZE.x + int(uv.y)*MAP_SIZE.x;
	
	return GetBit(charBitMap, index);
}

float DrawInt(int val, inout vec2 anchor, vec2 charSize, vec2 uv) {
	if (val == 0) return DrawChar(_0, anchor, charSize, uv);
	
	const int _DIGITS[10] = int[10](_0,_1,_2,_3,_4,_5,_6,_7,_8,_9);
	
	bool isNegative = val < 0.0;
	
	if (isNegative) string_to_draw[0] = _HYPH;
	
	val = abs(val);
	
	int posPlaces = int(ceil(log10(abs(val) + 0.001)));
	int strIndex = posPlaces - int(!isNegative);
	
	while (val > 0) {
		string_to_draw[strIndex--] = _DIGITS[val % 10];
		val /= 10;
	}
	
	return DrawString(anchor, charSize, posPlaces + int(isNegative), texcoord);
}

float DrawFloat(float val, inout vec2 anchor, vec2 charSize, int negPlaces, vec2 uv) {
	int whole = int(val);
	int part  = int(fract(abs(val)) * pow(10, negPlaces));
	
	int posPlaces = max(int(ceil(log10(abs(val)))), 1);
	
	anchor.x -= charSize.x * (posPlaces + int(val < 0) + 0.25);
	float ret = 0.0;
	ret += DrawInt(whole, anchor, charSize, uv);
	ret += DrawChar(_FSTP, anchor, charSize, texcoord);
	anchor.x -= charSize.x * 0.3;
	ret += DrawInt(part, anchor, charSize, uv);
	
	return ret;
}

void DrawDebugText() {
	vec2 charSize = vec2(0.05) * viewSize.yy / viewSize;
	vec2 texPos = vec2(charSize.x / 5.0, 1.0 - charSize.y * 1.2);
	
    int hideGUI = 0;

	if (hideGUI != 0
		|| texcoord.x > charSize.x * 8.0
		|| texcoord.y < 1 - charSize.y * 2.0)
	{ return; }
		
	vec3 color = vec3(0.0);
	float text = 0.0;
		
	//vec3 val = texelFetch(colortex7, ivec2(viewSize/2.0), 0).rgb;
    vec3 val = vec3(GetBeatFromTime(nonBlurTime), 0.0, 0.0);
    float bps = GetCameraPos(time + 0.01).z - GetCameraPos(time).z;
    val.r = bps / 0.01;
    val.r = GetCameraPos(time).z;
		
	string_to_draw = int[STRING_LENGTH](0,0, 0,0,0,0,0,0);
	text += DrawString(texPos, charSize, 2, texcoord);
	texPos.x += charSize.x * 2.0;
	text += DrawFloat(val.r, texPos, charSize, 2, texcoord);
	color += text * vec3(1.0, 1.0, 1.0) * sqrt(clamp(abs(val.r), 0.2, 1.0));
		
	texPos.x = charSize.x / 5.0, 1.0;
	texPos.y -= charSize.y * 1.4;
	
	OutColor.rgb = color;
}
/***********************************************************************/


/***********************************************************************/
/* Bloom */
vec4 cubic(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    vec4 w;
    
    w.x =   -x3 + 3*x2 - 3*x + 1;
    w.y =  3*x3 - 6*x2       + 4;
    w.z = -3*x3 + 3*x2 + 3*x + 1;
    w.w =  x3;
    
    return w / 6.0;
}

vec3 BicubicTexture(sampler2D tex, vec2 coord) {
    coord *= viewSize;
    
    vec2 f = fract(coord);
    
    coord -= f;
    
    vec4 xcubic = cubic(f.x);
    vec4 ycubic = cubic(f.y);
    
    vec4 c = coord.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    
    vec4 offset  = c + vec4(xcubic.yw, ycubic.yw) / s;
         offset /= viewSize.xxyy;
    
    vec3 sample0 = texture2D(tex, offset.xz).rgb;
    vec3 sample1 = texture2D(tex, offset.yz).rgb;
    vec3 sample2 = texture2D(tex, offset.xw).rgb;
    vec3 sample3 = texture2D(tex, offset.yw).rgb;
    
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    
    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

vec3 GetBloomTile(sampler2D tex, const int scale, vec2 offset) {
    vec2 coord  = texcoord;
         coord /= scale;
         coord += offset + 0.75/viewSize;
    
    return BicubicTexture(tex, coord);
}

#define BLOOM

#ifdef BLOOM
    const bool do_bloom = true;
#else
    const bool do_bloom = false;
#endif

#define BXLOOM_AMOUNT 0.4
#define BXLOOM_CURVE 1.0

vec3 GetBloom(sampler2D tex, vec3 color, vec2 outOffset) {
    if (!do_bloom)
        return color;
    
    vec3 bloom[8];
    
    // These arguments should be identical to those in composite2.fsh
    bloom[1] = GetBloomTile(tex,   4, vec2(0.0                          ,                           0.0) + outOffset);
    bloom[2] = GetBloomTile(tex,   8, vec2(0.0                          , 0.25     + 1/viewSize.y * 2.0) + outOffset);
    bloom[3] = GetBloomTile(tex,  16, vec2(0.125    + 1/viewSize.x * 2.0, 0.25     + 1/viewSize.y * 2.0) + outOffset);
    bloom[4] = GetBloomTile(tex,  32, vec2(0.1875   + 1/viewSize.x * 4.0, 0.25     + 1/viewSize.y * 2.0) + outOffset);
    bloom[5] = GetBloomTile(tex,  64, vec2(0.125    + 1/viewSize.x * 2.0, 0.3125   + 1/viewSize.y * 4.0) + outOffset);
    bloom[6] = GetBloomTile(tex, 128, vec2(0.140625 + 1/viewSize.x * 4.0, 0.3125   + 1/viewSize.y * 4.0) + outOffset);
    bloom[7] = GetBloomTile(tex, 256, vec2(0.125    + 1/viewSize.x * 2.0, 0.328125 + 1/viewSize.y * 6.0) + outOffset);
    
    bloom[0] = vec3(0.0);
    
    float totalWeight = 0.0;

    for (uint index = 1; index <= 7; index++) {
        float weight = index;
        bloom[0] += bloom[index] * weight;
        totalWeight += weight;
    }
    
    bloom[0] /= totalWeight;
    
    float bloom_amount = BXLOOM_AMOUNT;
    
    bloom[0] = max(bloom[0], 0.0);

    return bloom[0];

    return mix(color, min(pow(bloom[0], vec3(BXLOOM_CURVE)), bloom[0]), bloom_amount);
}
/***********************************************************************/

const mat3 ACESInputMat = mat3(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3 ACESOutputMat = mat3(
    1.60475, -0.53108, -0.07367,
    -0.10208, 1.10813, -0.00605,
    -0.00327, -0.07276, 1.07602
);

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 lottes(vec3 x) {
    const vec3 a = vec3(1.6);
    const vec3 d = vec3(0.977);
    const vec3 hdrMax = vec3(8.0);
    const vec3 midIn = vec3(0.18);
    const vec3 midOut = vec3(0.267);

    const vec3 b =
        (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    const vec3 c =
        (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(x, a) / (pow(x, a * d) * b + c);
}

vec3 GetColorLut(vec3 color) {
    //color.rgb = color.grb;
    color = clamp(color, 2.0 / 256.0, 1.0) * 254.0 / 256.0;
    vec2 texcoord = color.rg;
    int blue = int(color.b * 63);
    texcoord.x += blue % 8;
    texcoord.y += blue / 8;
    texcoord /= 8.0;
    texcoord.y = 1.0 - texcoord.y;
    return textureLod(lutTexture, texcoord, 0).rgb;
}

vec3 ACESFitted(vec3 color, bool sunspot) {
    color = max(color, 0.0);
    color = color * EXPOSURE;
    color = RRTAndODTFit(color * ACESInputMat) * ACESOutputMat;
    //color = aces(color);
    //color = lottes(color);
    color *= EXPOSURE2;
    //color = pow(color, vec3(1.0 / GAMMA));
    
    color = max(color, 0.0);

    if (!sunspot) color = log(color*1.0 + 1.0 + (vec3(0.08, 0.05, 0.1) - 0.05) / 8.0) / log(2.0);// * 1.5;
    color = pow(color, vec3(1.0 / GAMMA));
    
    //color = pow(color, vec3(1.4));
    //color = clamp(color, 0.0, 1.0);
    //color = hsv(color);
    // color.g = pow(color.g, 0.8);
    //color.g *= VIBRANCE;
    //color = rgb(color);
    //color = clamp(color, 0.0, 1.0);
    //color = GetColorLut(color);
    return color;
}

#define Screen(base, blend, a) mix(base, abs(1.0 - (blend - 1.0) * (base - 1.0)), a);

vec3 Method1() {
    vec2 cord = texcoord.xy * 1.0;
    if (texcoord.x + texcoord.y * 2.0 > 0.5) cord.x += 32.0 / textureSize(Texture13, 0).x;

    return texture(Texture13, cord).rgb;
}

vec3 Method2() {
    vec2 cord = texcoord.xy * 1.0;
    vec2 cord2 = cord;
    cord2 += 32.0 / textureSize(Texture13, 0).x;
    
    if (texcoord.x + texcoord.y * 2.0 > 0.5) return texture(Texture13, cord2).rgb;

    return texture(Texture13, cord).rgb;
}

void main() {
    OutColor = textureLod(Texture13, texcoord.xy, 0.0);

    if (OutColor.a < 0.0) { OutColor /= -OutColor.a; return; }

    vec3 bloom = GetBloom(bloomTexture, OutColor.rgb, vec2(0.0, 0.0)) / OutColor.a;

    vec3 sunBloom = (GetBloom(bloomTexture, OutColor.rgb, vec2(0.5, 0.0)) + textureLod(Texture9, texcoord.xy, 0.0).rgb) / OutColor.a;
    
    OutColor.rgb /= OutColor.a;
    
    OutColor.rgb = ACESFitted(OutColor.rgb, false);
    bloom = ACESFitted(bloom, false);
    bloom = clamp(bloom, 0.0, 1.0);
    OutColor.rgb = clamp(OutColor.rgb, 0.0, 1.0);
    bloom = rgb(clamp(hsv(bloom) * vec3(1,4,1), 0.0, 1.0));
    OutColor.rgb = Screen(OutColor.rgb, bloom, 0.15);

    sunBloom = ACESFitted(sunBloom, true);
    OutColor.rgb = Screen(OutColor.rgb, sunBloom, 1.0);

    //OutColor.rgb = vec3(OutColor.a != SAMPLE_COUNT) * 0.5;

    //OutColor = textureLod(Texture13, texcoord.xy, 0.0);
    //OutColor = textureLod(bloomTexture, texcoord.xy, 0.0);

    //OutColor = textureLod(bloomTexture, texcoord.xy, 0.0);

    //OutColor.rgb = Screen(ACESFitted(extra1), OutColor.rgb, 0.0);
    //OutColor.rgb = log(OutColor.rgb * 1.0 + 1.0 + (vec3(0.08, 0.05, 0.1) - 0.05) / 8.0) / log(2.0);// * 1.5;
    //OutColor.rgb = pow(OutColor.rgb, vec3(1.0 / GAMMA));

    //OutColor.rgb += vec3(GetVoxel(ivec3(gl_FragCoord.x, trackPos.y, gl_FragCoord.y)), 0.0);
    //OutColor.rgb += vec3(vec2(GetVoxel(ivec3(gl_FragCoord.x, trackPos.y, gl_FragCoord.y))), 0.0);
    //OutColor.rgb += vec3(vec2(VoxelBoolRead(ivec3(gl_FragCoord.x, trackPos.y, gl_FragCoord.y), 0)), 0.0);
    //OutColor.rgb += vec3(vec2(VoxelArray[int(gl_FragCoord.y*viewSize.x + gl_FragCoord.x)*16].x), 0.0);

    //OutColor.rgb += vec3(VoxelRead(ivec3(gl_FragCoord.x, trackPos.y, gl_FragCoord.y), 0));

    //float samp = simplex3d(vec3(texcoord * 10.0, texcoord.x * 1.0));
    //samp = texcoord.x * 2.0 - 1.0;
    //samp /= Gaussian(abs(samp), 0.3);
    //samp = NormalCDF(samp * 4.0);
    //OutColor.rgb = vec3(abs(samp) < 0.1);

    //OutColor.rgb = vec3(SparseChunkLoad(ivec3(gl_FragCoord.x/16, (WORLD_SIZE.y-1)/16, gl_FragCoord.y/16)).x != -1);
    //OutColor.rgb = vec3(chunkID[int(gl_FragCoord.x + gl_FragCoord.y * viewSize.x)].rgb / 100.0);
    

    //DrawDebugText();
    return;
};
#endif
