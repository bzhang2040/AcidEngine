#ifdef CXX_STAGE
#define FullscreenBlit_glsl "FullscreenBlit.glsl", "", "graphics"
#endif

#ifdef VERTEX_STAGE

layout(location = 0) in float _;

out vec2 position;

void main() {
    if (gl_VertexID == 0) gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
    if (gl_VertexID == 1) gl_Position = vec4(1.0, -1.0, 0.0, 1.0);
    if (gl_VertexID == 2) gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);
    if (gl_VertexID == 3) gl_Position = vec4(1.0, -1.0, 0.0, 1.0);
    if (gl_VertexID == 4) gl_Position = vec4(1.0, 1.0, 0.0, 1.0);
    if (gl_VertexID == 5) gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);

    position = gl_Position.xy;
};

#endif

#ifdef FRAGMENT_STAGE

layout(location = 0) out vec4 OutColor;

in vec2 position;

void main() {
    OutColor = vec4(position.xy * 0.5 + 0.5, 0.0, 1.0);
    OutColor = texture(frameTexture, position.xy * 0.5 + 0.5);
};

#endif
