{
    StartSpeed(80.0);
    Fisheye(1.0, -100, -99, temp);
    Shutter(1.0, -100, -99, temp);
    Acid(0.0, -100, -99, temp);
    SunAngle(45, -100, -99, temp);
    Fov(90.0, -100, -99, temp);
    Water(0.0, -100, -99, temp);
    Roll(0.0, -100, -99, temp);
    Pitch(0.001, -100, -99, temp);
    Yaw(89.999, -100, -99, temp);
    CameraHeight(WATER_HEIGHT+10.1, -100, -99, temp);
    
    CameraHeight(WATER_HEIGHT+5.0, 37, 50, EaseInOutSin(temp));
    CameraHeight(trackPos.y+2, 74, 96, EaseInOutSin(temp));
    
    Water(1.0, 49, 60, temp);
    Roll(1.0, 60, 84, EaseInOutSin(temp));
    Yaw(0.0, 36.9, 61, EaseOutSin(EaseInOutSin(temp)));
    
    Fisheye(0.0, 97, 145, temp);
    
    Shutter(0.5, 313-6, 313, temp);
    Fov(  120.0, 313-6, 313, tan(temp*3.14159/2.0/2.0));
    
    //  first chorus beat =  313
    // second chorus beat =  721
    //  third chorus beat = 1079
    
    Shutter(1.0, 360, 360+6, temp);
    Shutter(0.5, 1076-6, 1079, temp);
    
    Acid(0.2, 265, 271, powf(0.75, temp));
    Acid(0.6, 271, 275, powf(0.6, cubesmooth(temp)));
    Acid(0.8, 277, 313, temp);
    Acid(0.0, 500, 505, powf(4.0, temp));
    
    Fisheye(1.0, 600, 630, temp + 0.0*cubesmooth(tan(temp * 3.14159 / 4.0)));
    Fisheye(0.0, 700, 750, temp);
    
    SunAngle(175, 308, 505, temp);
    SunAngle(187, 505, 529, temp);
    SunAngle(354, 529, 673, temp);
    SunAngle(380, 673, 721, temp);
    
    Fov(110.0,   360, 360+3, tan(temp*3.14159/2.0/2.0));
    Fov(110.0,  1073,  1079, tan(temp*3.14159/2.0/2.0));
    
    Speed(500.0, 160, 160.1);
    Speed(80.0, 169, 169.1);

    Speed(160.0, 265, 275);
    Speed(120.0, 361, 366);

    Speed(500.0, 721-10, 721);
    Speed(120.0, 913, 937);

    Speed(160.0, 1073, 1079);
}
