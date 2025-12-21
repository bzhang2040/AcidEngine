#define id_both 0
#define id_left 1
#define id_right 2
bool CheckExtent(ivec3 pos, int x0, int x1, int y0, int y1, int leftright) {
    if (leftright == id_left) pos.x = -pos.x;

    if (leftright == id_both) pos.x = abs(pos.x);

    return pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1;
}

void B_(inout int val, ivec3 pos, bool GET_LIGHT, float z0) {
    if (pos.z == int(GetBeatPos(z0))) {
        if (GET_LIGHT || CheckExtent(pos, 1, 1, 2, 2, id_both)) val = id_torch;
        if (!GET_LIGHT && CheckExtent(pos, 1, 1, 0, 1, id_both)) val = id_permastone;
    }
}
#define B(z0) B_(val, pos, GET_LIGHT, z0);

void B_LOW_(inout int val, ivec3 pos, bool GET_LIGHT, float z0) {
    if (pos.z == int(GetBeatPos(z0))) {
        if (GET_LIGHT || CheckExtent(pos, 1, 1, 1, 1, id_both)) val = id_torch;
        if (!GET_LIGHT && CheckExtent(pos, 1, 1, 0, 0, id_both)) val = id_permastone;
    }
}
#define B_LOW(z0) B_LOW_(val, pos, GET_LIGHT, z0);

void B_WIDE_(inout int val, ivec3 pos, bool GET_LIGHT, float z0) {
    if (pos.z == int(GetBeatPos(z0))) {
        if (!GET_LIGHT && CheckExtent(pos, 2, 2, 2, 2, id_both)) val = id_permastone;
        if (GET_LIGHT || CheckExtent(pos, 1, 1, 2, 2, id_both)) val = pos.x > 0 ? id_torch_right : id_torch_left;
    }
}
#define B_WIDE(z0) B_WIDE_(val, pos, GET_LIGHT, z0);

void B_WIDE_L_(inout int val, ivec3 pos, bool GET_LIGHT, float z0) {
    if (pos.z == int(GetBeatPos(z0))) {
        if (!GET_LIGHT && (CheckExtent(pos, 2, 2, 2, 2, id_both) && pos.x < 0)) val = id_permastone;
        if (GET_LIGHT ||  (CheckExtent(pos, 1, 1, 2, 2, id_both) && pos.x < 0)) val = id_torch_left;
    }
}
#define B_WIDE_L(z0) B_WIDE_L_(val, pos, GET_LIGHT, z0);

void B_WIDE_R_(inout int val, ivec3 pos, bool GET_LIGHT, float z0) {
    if (pos.z == int(GetBeatPos(z0))) {
        if (!GET_LIGHT && (CheckExtent(pos, 2, 2, 2, 2, id_both) && pos.x > 0)) val = id_permastone;
        if (GET_LIGHT ||  (CheckExtent(pos, 1, 1, 2, 2, id_both) && pos.x > 0)) val = id_torch_right;
    }
}
#define B_WIDE_R(z0) B_WIDE_R_(val, pos, GET_LIGHT, z0);

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

float rad_to_turn(float rad) {
    return rad / (2.0 * 3.14159);
}

float turn_to_rad(float turn) {
    return turn * 2.0 * 3.14159;
}

bool inside(float x, float start, float end) {
    return x > start && x < end;
}

float atant(float x, float y) {
    return rad_to_turn(atan(x, y));
}

bool gt(vec2 x, vec2 y) {
    return all(greaterThan(x, y));
}

bool eq(ivec2 x, ivec2 y) {
    return all(greaterThan(x, y));
}

int AllBeats(ivec3 pos, bool GET_LIGHT) { float temp; int val = 0;
    // BEATS_TARGET
    return val;
}