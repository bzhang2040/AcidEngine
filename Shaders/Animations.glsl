
#define Key(a, b, c, d) temp = interp(beat, b, c); curr += d * NewValue(prev, a);
#define KeySpeed(a, b, c) temp = interpIntegral(GetTimeFromBeat(beat), GetTimeFromBeat(b), GetTimeFromBeat(c)); curr += temp * NewValue(prev, a);

#define powf(a, b) pow(b, a)

#define Fisheye(a, b, c, d)
#define Shutter(a, b, c, d)
#define Acid(a, b, c, d)
#define SunAngle(a, b, c, d)
#define SunRotation(a, b, c, d)
#define Fov(a, b, c, d)
#define Water(a, b, c, d)
#define Roll(a, b, c, d)
#define Pitch(a, b, c, d)
#define Yaw(a, b, c, d)
#define CameraHeight(a, b, c, d)
#define StartSpeed(a)
#define Speed(a, b, c)

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

float GetBeatPos(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef StartSpeed
    #define StartSpeed(a) prev = a; curr = GetTimeFromBeat(beat) * prev;
    #undef Speed
    #define Speed(a, b, c) KeySpeed(a, b, c)
    //#include "Animations2.glsl"
    #undef Speed
    #define Speed(a, b, c)
    #undef StartSpeed
    #define StartSpeed(a)
    return curr;
}

float GetWaterVisibility(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef Water
    #define Water(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Water
    #define Water(a, b, c, d)
    return curr;
}

float GetRoll(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef Roll
    #define Roll(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Roll
    #define Roll(a, b, c, d)
    return curr;
}

float GetPitch(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef Pitch
    #define Pitch(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Pitch
    #define Pitch(a, b, c, d)
    return radians(curr);
}

float GetYaw(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef Yaw
    #define Yaw(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Yaw
    #define Yaw(a, b, c, d)
    return radians(curr);
}

float GetCameraHeight(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    // Have to override beat here since this function is called before beatFromPos is set.
    // I believe it will be using the uniform data from the previous frame.
    beat = beatFromPos;
    
    #undef CameraHeight
    #define CameraHeight(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef CameraHeight
    #define CameraHeight(a, b, c, d)
    return curr;
}

vec3 GetCameraPos(float beat) {
    return vec3(trackPos.x, GetCameraHeight(beat), GetBeatPos(beat));
}

float ANIMATE_FOV(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef Fov
    #define Fov(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Fov
    #define Fov(a, b, c, d)
    return curr;
}

float SHUTTER_ANGLE(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef Shutter
    #define Shutter(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Shutter
    #define Shutter(a, b, c, d)
    return curr;
}

float GetSunAngle(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef SunAngle
    #define SunAngle(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef SunAngle
    #define SunAngle(a, b, c, d)
    return curr;
}

float GetSunRotation(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef SunRotation
    #define SunRotation(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef SunRotation
    #define SunRotation(a, b, c, d)
    return curr;
}

float FisheyeAmount(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    #undef Fisheye
    #define Fisheye(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Fisheye
    #define Fisheye(a, b, c, d)
    return curr;
}

float DistortionIntensity(float beat) { float prev = 0.0; float curr = 0.0; float temp = 0.0;
    if (!DO_DISTORTION) return 0.0;
    #undef Acid
    #define Acid(a, b, c, d) Key(a, b, c, d)
    //#include "Animations2.glsl"
    #undef Acid
    #define Acid(a, b, c, d)
    return curr;
}

vec3 TheFunction(vec3 pos) {
    vec3 oldPos = pos;
    vec3 offset = VoxelToWorld(vec3(0)) - cameraPosition * vec3(1,1,1) - trackPos.y * 0;
    pos += offset;
    oldPos = pos;

    {
        //pos.xy = rotate(-(pos.z / 100.0)) * pos.xy;

        //return pos;
    }

    if (distortionIntensity <= 0.0) return pos;

    //pos.y -= pos.z * pos.z / 1600.0;
    
    {
        pos.y += 2.0;
        
        float K = 3000.0 * (interp(beatFromPos, 265, 271));

        float t3 = (-pos.z * 3.0 / 200.0);
        return pos;
        t3 *= distortionIntensity;
        // t3 *= (interp(length(pos.xz), K, max(K - 500.0, 0.0)));
        // t3 *= mix(1.0, sin(baseFrameCameraPosition.z / 1000.0), interp(beatFromPos, 277, 300));
        pos.xy *= mat2(cos(t3), -sin(t3), sin(t3), cos(t3));
        // pos.xz *= rotate(radians(45.0));
        pos.xz *= rotate(radians(-60.0 * EaseInOutSin(interp(length(pos.xz), 0.0, 100.0))));
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
