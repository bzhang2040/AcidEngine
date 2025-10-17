
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

    return (x - 0.5) * b;
}

float NewValue(inout float value, float newValue) {
    float ret = (newValue - value);
    value = newValue;
    return ret;
}

float GetBeatPos(float beat) {
    float temp;
    float time = GetTimeFromBeat(beat);

    float prev = 80.0;
    float curr = time * prev;

    KeySpeed(500.0, 160, 160.1);
    KeySpeed(80.0, 169, 169.1);

    KeySpeed(160.0, 265, 275);
    KeySpeed(120.0, 361, 366);

    KeySpeed(500.0, 721-10, 721);
    KeySpeed(120.0, 913, 937);

    KeySpeed(160.0, 1073, 1079);

    return curr;
}

float GetLatent1() { float prev = 0.0; float curr = prev; float temp = 0.0; float beat = beatFromPos;
    Key(1.0, 49, 60, temp);
    return curr;
}

float GetLatent2() { float prev = 0.0; float curr = prev; float temp = 0.0; float beat = beatFromPos;
    Key(1.0, 60, 84, EaseInOutSin(temp));
    return curr;
}

float GetPitch() { float prev = -45.0; float curr = prev; float temp = 0.0; float beat = beatFromPos;
    
    Key(0, 60.9, 61, EaseInOutSin(temp));
    return radians(curr*0+0.001);

    return radians(mix(-45.0, 0.0, GetLatent2()));
}


vec3 GetCameraPos(float beat) { float prev = WATER_HEIGHT+1.1; float curr = prev; float temp = 0.0;
    float oldBeat = beat;
    beat = beatFromPos;

    Key(WATER_HEIGHT+5.0, 37, 50, EaseInOutSin(temp));
    Key(trackPos.y+2, 74, 96, EaseInOutSin(temp));

    return vec3(trackPos.x, (trackPos.y + curr)*0+curr, GetBeatPos(oldBeat));
}

float GetYaw() { float prev = 89.999; float curr = prev; float temp = 0.0; float beat = beatFromPos;
    //if (!writeFrames) return 0;

    Key(0.0, 36.9, 61, EaseOutSin(EaseInOutSin(temp)));

    return radians(curr);
}

float ANIMATE_FOV(float beat) {
    float var = 90.0;
    float old = var;

    var += NewValue(old, 120.0) * tan(interp(beat, 313-6, 313)*3.14159/2.0 / 2.0);
    var += NewValue(old, 110.0) * tan(interp(beat, 360, 360+3)*3.14159/2.0 / 2.0);

    var += NewValue(old, 110.0) * tan(interp(beat, 1073, 1079)*3.14159/2.0 / 2.0);

    return var;
}

float SHUTTER_ANGLE(float beat) {
    float prev = 1.0;
    float curr = prev;
    float temp = 0.0;

    Key(0.5, 313-6, 313, temp);
    Key(1.0, 360, 360+6, temp);
    Key(0.5, 1076-6, 1079, temp);

    return curr;
}

vec3 SunDirection(float beat) {
    //return normalize(vec3(0.2, 0.6, 0.3));

    float sunAngle = 45.0;
    float sunRotation = 30.0;

    float curr = 45.0;

    sunAngle += NewValue(curr, 175) * interp(beat, 308, 505);
    sunAngle += NewValue(curr, 187) * interp(beat, 505, 529);
    sunAngle += NewValue(curr, 354) * interp(beat, 529, 673);
    sunAngle += NewValue(curr, 380) * interp(beat, 673, 721);

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

float FisheyeAmount(float beat) {
    float prev = 1.0;
    float curr = prev;
    float temp = 0.0;

    Key(0.0, 97, 145, temp);

    Key(1.0, 600, 630, temp + 0.0*cubesmooth(tan(temp * 3.14159 / 4.0)));
    Key(0.0, 700, 750, temp);

    return curr;
}

float DistortionIntensity() {
    //return 1.0;
    if (!DO_DISTORTION) return 0.0;
    float prev = 0.0;
    float curr = 0.0;
    float temp = 0.0;
    float beat = beatFromPos;

    Key(0.2, 265, 271, powf(0.75, temp));
    Key(0.6, 271, 275, powf(0.6, cubesmooth(temp)));
    Key(0.8, 277, 313, temp);
    Key(0.0, 500, 505, powf(4.0, temp));

    return curr;
}

vec3 TheFunction(vec3 pos) {
    vec3 oldPos = pos;
    vec3 offset = VoxelToWorld(vec3(0)) - cameraPosition * vec3(1,1,1) - trackPos.y * 0;
    pos += offset;
    oldPos = pos;

    {
        //pos.xy = rotate(-(pos.z / 100.0)) * pos.xy;

        return pos;
    }

    if (distortionIntensity <= 0.0) return pos;

    //pos.y -= pos.z * pos.z / 1600.0;
    
    {
        pos.y += 2.0;
        
        float K = 3000.0 * (interp(beatFromPos, 265, 271));

        float t3 = sin(pos.z * 3.0 / currentSpeed);
        t3 *= distortionIntensity;
        t3 *= (interp(length(pos.xz), K, max(K - 500.0, 0.0)));
        t3 *= mix(1.0, sin(baseFrameCameraPosition.z / 1000.0), interp(beatFromPos, 277, 300));
        pos.xy *= mat2(cos(t3), -sin(t3), sin(t3), cos(t3));

        return pos;
    }

    {
        pos.y -= 10.0;
        pos.xz *= rotate(-yaw);
        pos.xy *= rotate(-mix(pos.x,pos.z,1.0) / 50.0);
        pos.xz *= rotate(yaw);

        return pos;
    }

    {
        pos.xz *= rotate(radians(20.0));
        pos.xy *= rotate(pos.z/50.0);
        pos.xz *= rotate(-radians(20.0));
        return mix(oldPos, pos, 0.6);
        return pos;
    }

    {
        float amt = cubesmooth(interp(beatFromPos, 16, 19));

        pos.xy += vec2(sin(pos.z / 10.0), cos(pos.z / 10.0)) * 10.0;
        pos.xy *= rotate(pos.z / 50.0);
        pos.xy -= vec2(sin(pos.z / 10.0), cos(pos.z / 10.0)) * 10.0;

        //return mix(oldPos, pos, amt);

        return pos;
    }

    return pos;
}
