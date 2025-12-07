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
    B(109) B(121) B(133);
    B_LOW(170) B(170.5) B(172) B_LOW(173) B(173.5) B(175) B_LOW(176) B(176.5);
    B(178) B(179) B(180) B(181) B(184); // I'll tell it to you one day
    B(190) B(190.5) B(191) B(192) B(193) B_LOW(194) B(194.5) B(196) B_LOW(197) B(197.5) B(199) B_LOW(200) B(200.5) B(202) B(203) B(204);
    B(205) B_LOW(206) B(206.5) B(208) B_LOW(209) B(209.5) B(211) B_LOW(212) B(212.5) B(214) B(215) B(216);
    B_LOW(218) B(218.5) B(220) B_LOW(221) B(221.5) B(223) B_LOW(224) B(224.5);
	B(226) B(227) B(228) B(229) B(232); // A mile on my one leg
	B(238) B(239) B(240) B(250) B(251) B(252) B(253) B_LOW(265); // Fixing my, fixing my eyes
    B_WIDE(271) B(277) B(278.5) B(280) B(281.5) B(283); // No way, you control my world
    B(284.5) B(286) B(287) B(288) B(289) B(292); // I'm on a straight line
    B(297.5) B(298) B(299) B(300); // The distant place
    B(309.5) B(310) B(311) B(312); // The distant way
    
    
    // Chorus 1
    if (pos.z >= int(GetBeatPos(313)) && pos.z < int(GetBeatPos(361))) {
        temp  = interp(pos.z, GetBeatPos(313), GetBeatPos(313.5));
        temp += interp(pos.z, GetBeatPos(313.5), GetBeatPos(314));
        temp += interp(pos.z, GetBeatPos(314), GetBeatPos(314.5));
        temp += interp(pos.z, GetBeatPos(314.5), GetBeatPos(315));
        
        int temp2 = int(floor(temp));
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            // if (GET_LIGHT || inside(atant(pos.x, pos.y), -0.25, 0.0) || inside(atant(pos.x, pos.y), 0.25, 0.5)) {
                if (int(pos.z) == int(GetBeatPos(313))) return id_beat;
                if (inside(int(pos.z), int(GetBeatPos(313)), int(GetBeatPos(313.5)))) return id_stone2;
            // }
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            // if (GET_LIGHT || inside(atant(pos.x, pos.y), -0.5, -0.25) || inside(atant(pos.x, pos.y), 0.0, 0.25)) {
                if (int(pos.z) == int(GetBeatPos(313.5))) return id_beat;
                if (inside(int(pos.z), int(GetBeatPos(313.5)), int(GetBeatPos(314)))) return id_stone2;
            // }
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            // if (GET_LIGHT || inside(atant(pos.x, pos.y-2), -0.5, -0.25) || inside(atant(pos.x, pos.y-2), 0.0, 0.25)) {
                if (int(pos.z) == int(GetBeatPos(314))) return id_beat;
                if (inside(int(pos.z), int(GetBeatPos(314)), int(GetBeatPos(314.5)))) return id_stone2;
            // }
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            // if (GET_LIGHT || inside(atant(pos.x, pos.y), -0.25, 0.0) || inside(atant(pos.x, pos.y), 0.25, 0.5)) {
                if (int(pos.z) == int(GetBeatPos(314.5))) return id_beat;
                if (inside(int(pos.z), int(GetBeatPos(314.5)), int(GetBeatPos(315)))) return id_stone2;
            // }
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            if (int(pos.z) == int(GetBeatPos(315))) return id_beat;
            if (inside(int(pos.z), int(GetBeatPos(315)), int(GetBeatPos(315.5)))) return id_stone2;
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            if (int(pos.z) == int(GetBeatPos(315.5))) return id_beat;
            if (inside(int(pos.z), int(GetBeatPos(315.5)), int(GetBeatPos(316)))) return id_stone2;
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            if (int(pos.z) == int(GetBeatPos(316))) return id_beat;
            if (inside(int(pos.z), int(GetBeatPos(316)), int(GetBeatPos(316.5)))) return id_stone2;
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            if (int(pos.z) == int(GetBeatPos(316.5))) return id_beat;
            if (inside(int(pos.z), int(GetBeatPos(316.5)), int(GetBeatPos(317)))) return id_stone2;
        }
        
        if (GET_LIGHT || (int(length(pos.xy+vec2(0,-2)))== 11)) {
            if (int(pos.z) == int(GetBeatPos(317))) return id_beat;
            if (inside(int(pos.z), int(GetBeatPos(317)), int(GetBeatPos(317.5)))) return id_stone2;
        }
        // {313}, {313.5}, {314}, {314.5}, {315}, {315.5}, {316}, {316.5}, {317}, {317.5}, {318}, {318.5}, {319}, {319.5}, {320}, {320.5}, {321}, {321.5}, {322}, {322.5}, {323},
        
        // vec3 crunched = crunch(position, vec3(1, 1, freq));
        // crunched.y += idx * 8.0;
        // float value = (simplex3d_fractal(crunched * vec3(1, 1, 0) / 16.0 / vec3(1, 0.25, 1)));
        // if (value > 0.4) return exact ? id_beat : id_stone2;
    }
    
    B(361) B(367) B(370) B(373); // You should know it's complicated
    B(385) B(391) B(394) B(397); // I'm all out of instigations
    B(409)B(409.25)B(409.5)B(410.5)B(410.75)B(411); // A spider on my wall
    
    B(412) B(412.25) B(412.5) B(413.5) B(413.75) B(414);
    B(415) B(415.25) B(415.5) B(416.5) B(416.75) B(417);
    B(418) B(418.25) B(418.5) B(419.5) B(419.75) B(420);
    
    B(420.5) B(421) B(421.25) B(421.5); // I let it start to crawl
    B(422.5) B(422.75) B(423) B(424) B(424.25) B(424.5);
    B(425.5) B(425.75) B(426) B(427) B(427.25) B(427.5);
    B(428.5) B(428.75) B(429);
    
    temp = interp(pos.z, GetBeatPos(411.1), GetBeatPos(420.0));
    if (Spiral(pos.xy, temp, 30.0, 10, 4, 2.0)) { return id_permastone; }
    
    B(430) B(433) B(434.5); // All over
    B(441) B(442) B(443) B(445) B(446.5) B(448) B(449.5);
    B(455.5);
    
    B(456.5) B(457) B(458.5) B(460) B(461.5) B(463); // I'm here but not for long
    B(468.5) B(469) B(470.5) B(472) B(473.5) B(475);
    B(478) B(479.5) B(481) B(482.5) B(490) B(491.5) B(493) B(494.5) B(496);
    
    // Sunset starts
    B_LOW(505);
    B_LOW(511) B_LOW(517) B_LOW(523) B_LOW(529) B_LOW(535) B_LOW(541) B_LOW(547);

    B_LOW(548) B_LOW(548.5) B_LOW(549.5) B_LOW(550);
    B_LOW(550.5) B_LOW(551) B_LOW(551.5) B_LOW(552) B_LOW(552.5);

    B_LOW(553) B_LOW(553.25) B_LOW(553.5);
    B_LOW(554.5) B_LOW(554.75) B_LOW(555);
    B_LOW(556) B_LOW(556.25) B_LOW(556.5);
    B_LOW(557.5) B_LOW(557.75) B_LOW(558);
    B_LOW(559) B_LOW(559.25) B_LOW(559.5);
    B_LOW(560.5) B_LOW(560.75) B_LOW(561);

    B_LOW(562) B_LOW(562.5) B_LOW(563) B_LOW(563.5) B_LOW(564) B_LOW(564.5) B_LOW(565) B_LOW(566.5) B_LOW(568);
    B_LOW(574) B_LOW(574.5) B_LOW(575) B_LOW(575.5) B_LOW(576) B_LOW(576.5) B_LOW(577) B_LOW(578.5);
    B_LOW(586) B_LOW(586.5) B_LOW(587) B_LOW(587.5) B_LOW(588) B_LOW(588.5) B_LOW(589) B_LOW(590.5);

    B_LOW(595) B_LOW(596.5) B_LOW(597.5) B_LOW(598) B_LOW(599) B_LOW(600);
    B_LOW(601) B_LOW(602) B_LOW(602.5);
    B_LOW(604) B_LOW(605) B_LOW(605.5);
    B_LOW(607) B_LOW(608) B_LOW(608.5);

    B_LOW(610) B_LOW(610.5) B_LOW(611) B_LOW(612) B_LOW(613) B_LOW(616); // Breaking down my heartache
    B_LOW(622) B_LOW(623) B_LOW(624);

    B_LOW(634) B_LOW(635) B_LOW(636) B_LOW(637);

    B_LOW(643) B_LOW(644.5) B_LOW(646) B_LOW(647.5) B_LOW(649);
    B_LOW(658) B_LOW(658.5) B_LOW(659) B_LOW(660) B_LOW(661) B_LOW(664);
    
    B_LOW(670) B_LOW(672);
    
    B_LOW(694) B_LOW(694.5) B_LOW(695) B_LOW(696) B_LOW(697) B_LOW(700); // Can it tell me always
    B_LOW(706) B_LOW(706.5) B_LOW(707) B_LOW(708) B_LOW(709) B_LOW(712);
    B_LOW(717.5) B_LOW(718) B_LOW(719) B_LOW(720);
    
    
    // Chorus 2
    B(721) B(721.5) B(722) B(722.5) B(723) B(723.5) B(724) B(724.5) B(725) B(725.5) B(726) B(726.5) B(727) B(727.5) B(728) B(728.5) B(729) B(729.5) B(730) B(730.5) B(731);
    B(733) B(733.5) B(734) B(734.5) B(735) B(735.5) B(736) B(736.5) B(737) B(737.5) B(738) B(738.5) B(739) B(739.5) B(740) B(740.5) B(741) B(741.5) B(742);
    B(745) B(745.5) B(746) B(746.5) B(747) B(747.5) B(748) B(748.5) B(749) B(749.5) B(750) B(750.5) B(751) B(751.5) B(752) B(752.5) B(753) B(753.5) B(754) B(754.5) B(755);
    B(757) B(757.5) B(758) B(758.5) B(759) B(759.5) B(760) B(760.5) B(761) B(761.5) B(762) B(762.5) B(763) B(763.5) B(764) B(764.5) B(765) B(765.5);
    
    B(766) B(766.5) B(767) B(767.5) B(768) B(768.5) B(769) B(770.5);// Every day a little closer
    
    B(774);
    
    return val;
}