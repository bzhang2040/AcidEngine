
StartSpeed(80.0);
Fisheye(1.0, -100, -99, temp);
Shutter(1.0, -100, -99, temp);
Distort(0.0, -100, -99, temp);
SunAngle(45, -100, -99, temp);
SunRotation(30, -100, -99, temp);
Fov(90.0, -100, -99, temp);
Water(0.0, -100, -99, temp);
Roll(0.0, -100, -99, temp);
Pitch(0.001, -100, -99, temp);
Yaw(89.999, -100, -99, temp);
CameraHeight(WATER_HEIGHT+1.1, -100, -99, temp);

Portal(0, WORLD_NAME(0));
Portal(160, WORLD_NAME(1));
Portal(169, WORLD_NAME(2));
Portal(217, WORLD_NAME(3));
Portal(1079, WORLD_NAME(4));

// TERRAIN_START
if (g_logicalWorldID == WORLD_NAME(0)) {
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
// TERRAIN_END
// TERRAIN_START
if (g_logicalWorldID == WORLD_NAME(1)) {
    return float(p.y > trackPos.y + 20.0);
}
// TERRAIN_END



// TERRAIN_START
if (g_logicalWorldID == WORLD_NAME(2)) {
    float sel = interp(Simplex(p, vec3(1024, 1e8, 1024), vec3(0)), 0.45, 0.55);
    vec2 sel2 = vec2(1.0 - sel, sel);
    sel2.x *= interp(p.y, 128, WATER_HEIGHT);
    sel2.y *= interp(p.y, 192, WATER_HEIGHT);
    float ret = 0.0;
    if (sel2.x > 0.0) ret += sel2.x * Simplex(p, vec3(256), vec3(0));
    if (sel2.y > 0.0) ret += sel2.y * Simplex(p, vec3(171), vec3(1e3));
    
    ret = mix(0.4, ret, interp(length(trackDist.xy), 0.0, 10.0));
    
    return ret;
}
// TERRAIN_END
// TERRAIN_START
if (g_logicalWorldID == WORLD_NAME(3)) {
    float v = mix(
        Simplex(p, vec3(256), vec3(0)) * interp(p.y, 256, WATER_HEIGHT),
        Simplex(p, vec3(256), vec3(1e3)) * interp(p.y, 256, WATER_HEIGHT),
        interp(Simplex(p, vec3(2048, 1e35, 2048), vec3(0)), 0.5, 0.52)
    );
    
    v = mix(v, 0.0, interp(length(trackDist.xy), 0.0, 10.0) * float((int(trackDelta.y))!=-1));
    
    return v;
}
// TERRAIN_END

Yaw(    0.0, 36.9, 61, EaseOutSin(EaseInOutSin(temp)));
Water(  1.0, 49, 60, temp);
Roll(   1.0, 60, 84, EaseInOutSin(temp));
Fisheye(0.0, 97, 145, temp);
CameraHeight(WATER_HEIGHT+5.0, 37, 50, EaseInOutSin(temp));
CameraHeight(trackPos.y+2, 74, 96, EaseInOutSin(temp));

    { // BEATS_START
        B(109) B(121) B(133);
    } // BEATS_END

Speed(500.0, 160, 160.1);
Speed( 80.0, 169, 169.1);

    { // BEATS_START
        B_LOW(170) B(170.5) B(172) B_LOW(173) B(173.5) B(175) B_LOW(176) B(176.5);
        B(178) B(179) B(180) B(181) B(184); // I'll tell it to you one day
        B(190) B(190.5) B(191) B(192) B(193) B_LOW(194) B(194.5) B(196) B_LOW(197) B(197.5) B(199) B_LOW(200) B(200.5) B(202) B(203) B(204);
        B(205) B_LOW(206) B(206.5) B(208) B_LOW(209) B(209.5) B(211) B_LOW(212) B(212.5) B(214) B(215) B(216);
        B_LOW(218) B(218.5) B(220) B_LOW(221) B(221.5) B(223) B_LOW(224) B(224.5);
        B(226) B(227) B(228) B(229) B(232); // A mile on my one leg
        B(238) B(239) B(240) B(250) B(251) B(252) B(253) B_LOW(265); // Fixing my, fixing my eyes
    } // BEATS_END

Speed( 320.0, 265, 275);

Distort(0.2, 265, 271, powf(0.75, temp));

    { // BEATS_START
        B_WIDE(271) B(277) B(278.5) B(280) B(281.5) B(283); // No way, you control my world
    } // BEATS_END

Distort(0.6, 271, 275, powf(0.6, cubesmooth(temp)));
Distort(0.8, 277, 313, temp);

    { // BEATS_START
        B(284.5) B(286) B(287) B(288) B(289) B(292); // I'm on a straight line
        B(297.5) B(298) B(299) B(300); // The distant place
        B(309.5) B(310) B(311) B(312); // The distant way
    } // BEATS_END


Shutter( 0.5, 307, 313, temp);
Fov(   120.0, 307, 313, cubesmooth(temp*3.14159/2.0/2.0));
SunAngle(175, 308, 505, temp);

    // ----------------------------------
    // ----- First chorus beat: 313 -----
    // ----------------------------------
    
    
    // ACID_START
    {
        pos.y += 2.0;
        float t3 = (-pos.z * 3.0 / 200.0);
        t3 *= distortionIntensity;
        pos.xy *= mat2(cos(t3), -sin(t3), sin(t3), cos(t3));
        return pos;
    }
    // ACID_END
    

    // BEATS_START
    if (pos.z >= int(GetBeatPos(313)) && pos.z < int(GetBeatPos(361))) {
        /*
        {.b=313, .d=0.0/30}, {313.5}, {314}, {314.5}, {315}, {315.5}, {316}, {316.5}, {317}, {317.5}, {318}, {318.5}, {319}, {319.5}, {320}, {320.5}, {321}, {321.5}, {322}, {322.5}, {323},
        {325}, {325.5}, {326}, {326.5}, {327}, {327.5}, {328}, {328.5}, {329}, {329.5}, {330}, {330.5}, {331}, {331.5}, {332}, {332.5}, {333}, {333.5}, {334},
        {337}, {337.5}, {338}, {338.5}, {339}, {339.5}, {340}, {340.5}, {341}, {341.5}, {342}, {342.5}, {343}, {343.5}, {344}, {344.5}, {345}, {345.5}, {346}, {346.5}, {347},
        {349}, {349.5}, {350}, {350.5}, {351}, {351.5}, {352}, {352.5}, {353}, {353.5}, {354}, {354.5}, {355}, {355.5}, {356}, {356.5}, {357}, {357.5}, {358}, {358.5}, {359}, {359.5}, {360}, {360.5},
        */
        
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
    // BEATS_END

Shutter(1.0, 360, 366, temp);
Fov(  90.0, 360, 363, cubesmooth(cubesmooth(temp*3.14159/2.0/2.0)));
Speed(120.0, 361, 366);

    { // BEATS_START
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
        
        B_LOW(505);
    } // BEATS_END



// ----- Sunset starts -----
// Distort(0.0, 500, 505, powf(4.0, temp));
SunAngle(187, 505, 529, temp);
SunAngle(354, 529, 673, temp);

    { // BEATS_START
        B_LOW(511) B_LOW(517) B_LOW(523) B_LOW(529) B_LOW(535) B_LOW(541) B_LOW(547);
        
        B_LOW(548) B_LOW(548.5) B_LOW(549.5) B_LOW(550);
        B_LOW(550.5) B_LOW(551) B_LOW(551.5) B_LOW(552) B_LOW(552.5);
        // B_LOW(553) B_LOW(553.25) B_LOW(553.5);
        
        if (pos.z == int(GetBeatPos(553))) {
            if (GET_LIGHT || CheckExtent(pos, -2, -2, 3, 3, id_left)) val = id_glowstone;
            // if (!GET_LIGHT && CheckExtent(pos, 1, 1, 0, 0, id_both)) val = id_permastone;
        }
        
        if (pos.z == int(GetBeatPos(553.25))) {
            if (GET_LIGHT || CheckExtent(pos, 0, 0, 4, 4, id_left)) val = id_glowstone;
            // if (!GET_LIGHT && CheckExtent(pos, 1, 1, 0, 0, id_both)) val = id_permastone;
        }
        
        if (pos.z == int(GetBeatPos(553.5))) {
            if (GET_LIGHT || CheckExtent(pos, 2, 2, 3, 3, id_left)) val = id_glowstone;
            // if (!GET_LIGHT && CheckExtent(pos, 1, 1, 0, 0, id_both)) val = id_permastone;
        }
        
        B_LOW(601) B_LOW(602) B_LOW(602.5);
        B_LOW(604) B_LOW(605) B_LOW(605.5);
        B_LOW(607) B_LOW(608) B_LOW(608.5);

        B_LOW(610) B_LOW(610.5) B_LOW(611) B_LOW(612) B_LOW(613) B_LOW(616); // Breaking down my heartache
        B_LOW(622) B_LOW(623) B_LOW(624);

        B_LOW(634) B_LOW(635) B_LOW(636) B_LOW(637);

        B_LOW(643) B_LOW(644.5) B_LOW(646) B_LOW(647.5) B_LOW(649);
        B_LOW(658) B_LOW(658.5) B_LOW(659) B_LOW(660) B_LOW(661) B_LOW(664);
        
        B_LOW(670) B_LOW(672);
    } // BEATS_END

Fisheye( 1.0, 600, 630, temp + 0.0*cubesmooth(tan(temp * 3.14159 / 4.0)));
SunAngle(380, 673, 721, temp);
SunAngle(360+120, 722, 723, temp);
Fisheye( 0.0, 700, 750, temp);

    { // BEATS_START
        B_LOW(694) B_LOW(694.5) B_LOW(695) B_LOW(696) B_LOW(697) B_LOW(700); // Can it tell me always
        B_LOW(706) B_LOW(706.5) B_LOW(707) B_LOW(708) B_LOW(709) B_LOW(712);
        B_LOW(717.5) B_LOW(718) B_LOW(719) B_LOW(720);
    } // BEATS_END

// -----------------------------------
// ----- Second chorus beat: 721 -----
// -----------------------------------

    { // BEATS_START
        B(721) B(721.5) B(722) B(722.5) B(723) B(723.5) B(724) B(724.5) B(725) B(725.5) B(726) B(726.5) B(727) B(727.5) B(728) B(728.5) B(729) B(729.5) B(730) B(730.5) B(731);
        B(733) B(733.5) B(734) B(734.5) B(735) B(735.5) B(736) B(736.5) B(737) B(737.5) B(738) B(738.5) B(739) B(739.5) B(740) B(740.5) B(741) B(741.5) B(742);
        B(745) B(745.5) B(746) B(746.5) B(747) B(747.5) B(748) B(748.5) B(749) B(749.5) B(750) B(750.5) B(751) B(751.5) B(752) B(752.5) B(753) B(753.5) B(754) B(754.5) B(755);
        B(757) B(757.5) B(758) B(758.5) B(759) B(759.5) B(760) B(760.5) B(761) B(761.5) B(762) B(762.5) B(763) B(763.5) B(764) B(764.5) B(765) B(765.5);
        
        B(766) B(766.5) B(767) B(767.5) B(768) B(768.5) B(769) B(770.5);// Every day a little closer
        
        B(774);
        
        B_WIDE_R(775) B_WIDE_L(775.5) B_WIDE_R(776) B_WIDE_L(776.5) B_WIDE_R(777) B_WIDE_L(777.5);
        B_WIDE_R(778) B_WIDE_L(778.5) B_WIDE_R(779) B_WIDE_L(779.5) B_WIDE_R(780) B_WIDE_L(780.5);
        
        B(781);
        B_LOW(782.5) B_LOW(784);
    } // BEATS_END

// Speed(500.0, 711, 721);
Speed(120.0, 913, 937);

Shutter(0.5, 1070, 1079, temp);
Speed(160.0, 1073, 1079);
Fov(  110.0, 1073, 1079, cubesmooth(temp*3.14159/2.0/2.0));


// -----------------------------------
// ----- Third chorus beat: 1079 -----
// -----------------------------------
