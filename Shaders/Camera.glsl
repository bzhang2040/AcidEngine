

const vec2 trackPos = vec2(0.5, 92.5);

const float customyaw = 0.0000001;
const float custompitch = -0.001;
const float customzoom = 0.0;
const vec3 custommovement = vec3(0.0, 0.0, 0.0) * vec3(1.0) + vec3(0.0001);

#define VIDEO_LENGTH_SECONDS (9*60)

#define BLOCKS_PER_SECOND 80.0f
#define BEATS_PER_MINUTE 160.0f
#define BEATS_PER_SECOND (BEATS_PER_MINUTE / 60.0f)

#define Key(a, b, c, d) temp = interp(beat, b, c); curr += d * NewValue(prev, a);
#define KeySpeed(a, b, c) temp = interpIntegral(GetTimeFromBeat(beat), GetTimeFromBeat(b), GetTimeFromBeat(c)); curr += temp * NewValue(prev, a);

#define powf(a, b) pow(b, a)

float interp(float x, float a, float b) {
    if (b == a) return 0.0;
    if (b > a) return clamp((x - a) / (b - a), 0.0f, 1.0f);
    return clamp((x - a) / (b - a), 0.0f, 1.0f);
}

float interpIntegral(float x, float a, float b) {
    x -= a;
    b -= a;

    if (x < 0.0) return 0.0;

    x /= b;

    if (x < 1.0) {
        return 0.5 * x * x * b;
    }

	return (x - 0.5)*b;
}

float cubesmooth2(float x) {
    return x * x * (3.0 - 2.0 * x);
}

float GetBeatFromTime(float time) {
    float secondsPerBeat = 60.0 / BEATS_PER_MINUTE;

    return time / secondsPerBeat;
}

float GetTimeFromBeat(float beat) {
    float secondsPerBeat = 60.0 / BEATS_PER_MINUTE;

    return beat * secondsPerBeat;
}

#ifdef CXX_STAGE
float NewValue(float& value, float newValue) {
    float ret = (newValue - value);
    value = newValue;
    return ret;
}
#else
float NewValue(inout float value, float newValue) {
    float ret = (newValue - value);
    value = newValue;
    return ret;
}
#endif

vec3 GetCameraPos(float time) {
    float temp;
    float beat = GetBeatFromTime(time);
    
    float prev = 80.0;
    float curr = time * prev;


    KeySpeed(500.0, 160, 160+0.1);
    KeySpeed(80.0, 169, 169+0.1);

    KeySpeed(160.0, 265, 275);
    KeySpeed(120.0, 361, 366);

    KeySpeed(160.0, 1073, 1079);
    
    return vec3(trackPos.x, trackPos.y + 2.0, curr);
}

float GetBeatPos(float beat) {
    float secondsPerBeat = 60.0 / BEATS_PER_MINUTE;

    return GetCameraPos(secondsPerBeat * beat).z;
}

#if !defined(CXX_STAGE)
float ANIMATE_FOV(float time) {
    float var = 90.0;
    float old = var;

    var += NewValue(old, 120.0) * tan(interp(time, GetTimeFromBeat(313-6), GetTimeFromBeat(313))*3.14159/2.0 / 2.0);
    var += NewValue(old, 90.0) * tan(interp(time, GetTimeFromBeat(360), GetTimeFromBeat(360+3))*3.14159/2.0 / 2.0);

    var += NewValue(old, 110.0) * tan(interp(time, GetTimeFromBeat(1073), GetTimeFromBeat(1079))*3.14159/2.0 / 2.0);

    return var;
}

mat2 rotate2(float rad) {
    return mat2(cos(-rad), -sin(-rad), sin(-rad), cos(-rad));
}

float SHUTTER_ANGLE(float time) {
    float prev = 1.0;
    float curr = prev;
    float temp = 0.0;
    float beat = GetBeatFromTime(time);

    Key(0.5, 313-6, 313, temp);
    Key(1.0, 360, 360+6, temp);
    Key(0.5, 1076-6, 1079, temp);

    return curr;
}

vec3 SunDirection(float zPos) {
    //return normalize(vec3(0.2, 0.6, 0.3));

    float sunAngle = 45.0;
    float sunRotation = 30.0;

    float curr = 45.0;

    sunAngle += NewValue(curr, 175) * interp(zPos, GetCameraPos(GetTimeFromBeat(308)).z, GetCameraPos(GetTimeFromBeat(505)).z);
    sunAngle += NewValue(curr, 187) * interp(zPos, GetCameraPos(GetTimeFromBeat(505)).z, GetCameraPos(GetTimeFromBeat(529)).z);
    sunAngle += NewValue(curr, 354) * interp(zPos, GetCameraPos(GetTimeFromBeat(529)).z, GetCameraPos(GetTimeFromBeat(673)).z);
    sunAngle += NewValue(curr, 380) * interp(zPos, GetCameraPos(GetTimeFromBeat(673)).z, GetCameraPos(GetTimeFromBeat(721)).z);

    vec3 sunDir = vec3(0.0, 0.0, 1.0);

    vec2 v = vec2(sunDir.x, sunDir.z) * rotate2(radians(sunRotation));
    sunDir.x = v.x;
    sunDir.z = v.y;
    v = vec2(sunDir.y, sunDir.z) * rotate2(radians(sunAngle));
    sunDir.y = v.x;
    sunDir.z = v.y;

    return normalize(sunDir);
}

vec3 MoonDirection(float zPos) {
    vec3 moonDirection = SunDirection(zPos);
    vec2 v = vec2(moonDirection.x, moonDirection.z) * rotate2(radians(-30.0));
    moonDirection.x = v.x;
    moonDirection.z = v.y;
    moonDirection.y *= -1.0;
    moonDirection = -SunDirection(zPos);

    return moonDirection;
}

float FisheyeAmount(float time) {
    float prev = 0.0;
    float curr = 0.0;
    float temp = 0.0;
    float beat = GetBeatFromTime(time);
    return 1.0;
    //Key(1.0, 650, 670, cubesmooth2(tan(temp * 3.14159 / 4.0)));
    //Key(1.0, 665, 670, atan(temp)/3.14159*4.0);
    //Key(1.0, 650, 670, cubesmooth2(temp));

    //curr += NewValue(prev, 1.0) * tan(interp(time, GetTimeFromBeat(312), GetTimeFromBeat(313)) * 3.14159 / 2.0 / 2.0);

    return curr;
}

float GetTimeFromPos(float pos) { // Inverts the function using newtons method.
    float t = 0.0;
    
    for (int i = 0; i < 100; ++i) {
        float f = GetCameraPos(t).z - pos;
        float df = GetCameraPos(t + 1.0).z - GetCameraPos(t).z;
        
        if (abs(f) < 0.0001 || df == 0.0) {
            return t;
        }

        t -= f / df;
    }

    return t;
}

#endif
