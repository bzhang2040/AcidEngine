vec3 position = p;
switch(g_logicalWorldID) {
    case WORLD_NAME(0): {
        return
        mix(
            p.y<trackPos.y+40 ? 0.85*Simplex(p+vec3(0,0,0), vec3(1e16, 16, 1e16), vec3(0))
                                : 0.0,
            0.85*Simplex(p, vec3(256), vec3(0)),
            0.5)
            //*mix(0.7,1.0, interp(distance(p.z,GetBeatPos(28.2)+5),0,10))
            *mix(0.0,1.0, interp(distance(p.xz,vec2(trackPos.x,GetBeatPos(49)+50)),0,100))
            //*mix(0.5,1.0,interp(length(trackDist),0,10))
            *mix(0.8,1.0,interp(trackDist.x,0,10))
            //*mix(0.5,1.0,interp(trackDist.y,0,10))
            ;
    }
    case WORLD_NAME(1): {
        return float(p.y > trackPos.y + 20.0);
    }
    case WORLD_NAME(2): {
        float sel = interp(Simplex(p, vec3(1024, 1e8, 1024), vec3(0)), 0.45, 0.55);
        vec2 sel2 = vec2(1.0 - sel, sel);
        sel2.x *= interp(p.y, 128, WATER_HEIGHT);
        sel2.y *= interp(p.y, 192, WATER_HEIGHT);
        float ret = 0.0;
        if (sel2.x > 0.0) ret += sel2.x * Simplex(p, vec3(256), vec3(0));
        if (sel2.y > 0.0) ret += sel2.y * Simplex(p, vec3(171), vec3(1e3));
        return ret;
    }
    case WORLD_NAME(3): {
        p.z = floor(p.z / 100.0) * 100.0;
        p.xy -= trackPos.xy;
        p.xy = rotate(p.z / 100.0) * p.xy;
        p.xy += trackPos.xy;
        
        float v = Simplex(p, vec3(171) * vec3(1, 1, 1), vec3(0));

        v *= interp(length(trackDist.xy), 1000.0, 0.0);
        v *= mix(interp(-(p.x - trackPos.x), 0.0, 100.0), 1.0, 0.5);

        return v;
    }
}
return 0.0;
