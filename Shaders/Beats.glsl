#define id_both 0
#define id_left 1
#define id_right 2
bool CheckExtent(ivec3 pos, int x0, int x1, int y0, int y1, int leftright) {
    if (leftright == id_left) pos.x = -pos.x;

    if (leftright == id_both) pos.x = abs(pos.x);

    return pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1;
}

#define B(z0) \
    if (pos.z == int(GetBeatPos(z0))) { \
        if (GET_LIGHT || CheckExtent(pos, 1, 1, 2, 2, id_both)) return id_torch; \
        if (!GET_LIGHT && CheckExtent(pos, 1, 1, 0, 1, id_both)) return id_permastone; \
    }

#define B_LOW(z0) \
    if (pos.z == int(GetBeatPos(z0))) { \
        if (GET_LIGHT || CheckExtent(pos, 1, 1, 1, 1, id_both)) return id_torch; \
        if (!GET_LIGHT && CheckExtent(pos, 1, 1, 0, 0, id_both)) return id_permastone; \
    }
    
#define DEFAULT_BEAT2(x0, x1, y0, y1, z0, z1, side, blockType) \
    if (pos.z >= int(GetBeatPos(z0)) && pos.z >= int(GetBeatPos(z0)) && CheckExtent(pos, x0, x1, y0, y1, side)) { return blockType; }

    
bool Spiral(vec2 pos, float temp, float period, int radius) {
    if (temp <= 0.0 || temp >= 1.0) return false;
    temp *= period;
    return (int(pos.x) == int(sin(temp) * radius) && int(pos.y) == int(cos(temp) * radius));
}

bool Spiral(vec2 pos, float temp, float period, int radius, int branches) {
    for (int i = 0; i < branches; ++i) {
        if (Spiral(pos, temp+float(i)/branches / period*3.14159*2.0, period, radius)) return true;
    }
    
    return false;
}

bool Spiral(vec2 pos, float temp, float period, int radius, int branches, float width) {
    if (temp <= 0.0 || temp >= 1.0) return false;
    float theta = atan(pos.y, pos.x);
    float r = length(pos.xy);
    float theta_spiral = temp * period;
    float sector = round( (theta - theta_spiral) / (2*3.14159 / branches) );
    float theta_expected = theta_spiral + sector * (2*3.14159 / branches);
    return (int(r) == radius && abs(theta - theta_expected) < width/radius);
}

int AllBeats(ivec3 pos, bool GET_LIGHT) { float temp;
    B(109) B(121) B(133);
    B_LOW(170) B(170.5) B(172) B_LOW(173) B(173.5) B(175) B_LOW(176) B(176.5);
    B(178) B(179) B(180) B(181) B(184); // I'll tell it to you one day
    B(190) B(190.5) B(191) B(192) B(193) B_LOW(194) B(194.5) B(196) B_LOW(197) B(197.5) B(199) B_LOW(200) B(200.5) B(202) B(203) B(204);
    B(205) B_LOW(206) B(206.5) B(208) B_LOW(209) B(209.5) B(211) B_LOW(212) B(212.5) B(214) B(215) B(216);
    B_LOW(218) B(218.5) B(220) B_LOW(221) B(221.5) B(223) B_LOW(224) B(224.5);
	B(226) B(227) B(228) B(229) B(232); // A mile on my one leg
	B(238) B(239) B(240) B(250) B(251) B(252) B(253) B_LOW(265); // Fixing my, fixing my eyes
    
    B(409)B(409.25)B(409.5)B(410.5)B(410.75)B(411);
    
    temp = interp(pos.z, GetBeatPos(411.1), GetBeatPos(420.0));
    if (Spiral(pos.xy, temp, 30.0, 10, 4, 2.0)) { return id_permastone; }
    
    return 0;
}