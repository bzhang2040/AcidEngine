#include <string>

#include "glad.h"
#include "glfw3.h"
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/common.hpp>

#include <unordered_map>
#include <thread>
#include <filesystem>
#include <format>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image.h"
#include "stb_image_write.h"

using namespace glm;
using std::vector, std::thread, std::unordered_map, std::cout, std::fstream, std::string;

#include "BS.h"

static bool doubleBuffer = true;
static bool vSync = false;
static bool cleanFramesFolder = true;

float maxSeconds = 30.0; // 485.0;
float framerate = 60.0;

#define FOV 90.0f

#define TIME_OFFSET (-0.375)

GLFWwindow* window;
float camyaw = 0.0f;
float campitch = 0.0f;
vec2 camVelocity = vec2(0.0);
vec4 camMoments = vec4(4.0, 0.00, 0.0, 1.00);

vec3 posVelocity = vec3(0.0);
vec4 displaceMoments = vec4(1.0, 0.00, 0.0, 0.00);


float zoom = 0.0f;

bool KeysPressed[512] = { false };
bool KeyPressEvent(int keyID) {
    if (glfwGetKey(window, keyID) == GLFW_PRESS) {
        bool result = !KeysPressed[keyID];
        KeysPressed[keyID] = true;
        return result;
    }
    else {
        KeysPressed[keyID] = false;
        return false;
    }
}

bool KeysToggled[512] = { false };
bool KeyIsToggled(int keyID) {
    bool temp = KeysToggled[keyID];

    if (KeyPressEvent(keyID)) {
        KeysToggled[keyID] = !KeysToggled[keyID];
    }

    return temp;
}

int skyLutSize = 256 * 128 * 33 * 4 * 2;
char skyLutData[256 * 128 * 33 * 4 * 2 * 2];

// Terrain generation
Shader computeProg(TriangleInit_glsl);
Shader initChunks0(InitChunks0_glsl);
Shader initChunks(InitChunks_glsl);
Shader computeChunkUpdates(ComputeChunkUpdates_glsl);
Shader computeDense(ComputeDense_glsl);
Shader deallocChunks(DeallocChunks_glsl);
Shader clearLOD(ClearLod_glsl);
Shader topsoil(Topsoil_glsl);
Shader generateLOD(GenerateLOD_glsl);

// Rendering
Shader initBeatStruct(InitBeats_glsl);
Shader uniformShader(Uniforms_glsl);
Shader composite0(Composite0_glsl);
Shader bloom(Bloom_glsl);
Shader tonemap(Tonemap_glsl);

void Dispatch(int program, int VAO) {
    glBindVertexArray(VAO);
    glUseProgram(program);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

class SSBO {
public:
    GLuint id;
    int binding;
    int size;

    SSBO() : binding(0), size(0) {}

    SSBO(int binding_, int size_) : binding(binding_), size(size_) {
        glCreateBuffers(1, &id);
        glNamedBufferStorage(id, size * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id);
    }

    void Init(int size_) {
        size = size_;

        glCreateBuffers(1, &id);
        glNamedBufferStorage(id, size * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
    }

    void Bind(int binding_) {
        binding = binding_;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id);
    }
};

class Terrain {
public:
    void Init() {
        if (initialized) {
            return;
        }

        initialized = true;

        voxelData = new Texture(GL_R8UI, sparseChunkDims.x*16, ((sparseChunkDims.y*5)/4)*16, sparseChunkDims.z*16, "nearest", nullptr);

        chunkData = new Texture(GL_RGBA16I, denseChunkDims.x, denseChunkDims.y*MAX_WORLD_COUNT, denseChunkDims.z, "nearest", nullptr);
        skyLUT = new Texture(GL_RGBA16F, 256, 128, 33, "linear", skyLutData);

        computeIndirectCount.Init(3 * 4);
        chunkIndirectUpdates.Init(denseChunkDims.x*denseChunkDims.y*denseChunkDims.z * 4);
        chunkIDs.Init((1024 + sparseTotalSize*4));
    }

    void Bind() {
        voxelData->BindImage(0);
        chunkData->BindImage(1);
        skyLUT->Bind(1, GL_TEXTURE_3D);
        computeIndirectCount.Bind(2);
        chunkIndirectUpdates.Bind(5);
        chunkIDs.Bind(3);
        // There is only 1 active indirect buffer, with slot 0
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, computeIndirectCount.id);
    }

    void Regenerate(bool reload) {
        ivec3 dispatchSize = WORLD_SIZE / ivec3(16, 16, 16);
        
        Dispatch(computeProg, 1, 1, 1);

        if (UPDATE_INDIRECT) Dispatch(computeChunkUpdates, WORLD_SIZE.x / 16 / 16, WORLD_SIZE.y / 16, WORLD_SIZE.z / 16 / 16);
        DispatchIndirect(clearLOD, chunkUpdates * 16);
        if (reload) Dispatch(initChunks0, denseChunkDims.x/16, denseChunkDims.y, denseChunkDims.z/16);
        if (reload) Dispatch(initChunks, sparseChunkDims.x/16, sparseChunkDims.y, sparseChunkDims.z/16);
        if (!reload) Dispatch(deallocChunks, denseChunkDims.x/16, denseChunkDims.y, denseChunkDims.z/16); // One thread per-chunk
        DispatchIndirect(computeDense, chunkUpdates*16); // One threadblock per-chunk
        DispatchIndirect(topsoil, chunkUpdates*16);
        DispatchIndirect(generateLOD, chunkUpdates*16);
    }

    Texture* voxelData;
    Texture* chunkData;
    Texture* skyLUT;
    SSBO chunkIDs;
    SSBO chunkIndirectUpdates;
    SSBO computeIndirectCount;
    bool initialized = false;
    int shaderIncrement = ShaderIncrement;
};

void ProcessInput(vec3& movement, float samples) {
    float multiplier = 1.0f;

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) multiplier *= 0.1f;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) multiplier *= 3.0f;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) multiplier *= 10.0f;

    vec3 offset = vec3(0.0);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) offset.x += multiplier;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) offset.x -= multiplier;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) offset.z += multiplier;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) offset.z -= multiplier;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) offset.y += multiplier;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) offset.y -= multiplier;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) offset.y -= multiplier;

    camyaw += camVelocity.x;
    campitch += camVelocity.y;
    campitch = glm::max(glm::min(campitch, 3.14159f / 2.0f), -3.14159f / 2.0f);

    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) camyaw -= 0.07f * glm::min(multiplier, 1.0f);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) camyaw += 0.07f * glm::min(multiplier, 1.0f);

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) campitch = glm::max(glm::min(campitch - 0.07f * glm::min(multiplier, 1.0f), 3.14159f / 2.0f), -3.14159f / 2.0f);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) campitch = glm::max(glm::min(campitch + 0.07f * glm::min(multiplier, 1.0f), 3.14159f / 2.0f), -3.14159f / 2.0f);

    offset = vec3(glm::rotate(glm::mat4(1.0f), camyaw + customyaw, vec3(0.0, 1.0, 0.0)) * vec4(offset.x, offset.y, offset.z, 0.0f));

    posVelocity += offset;
    movement += offset * displaceMoments[0] + posVelocity * displaceMoments[1];
}

float floor16(float x) {
    return x - mod(x, 16.0f);
}

bool diff16(float x, float y) {
    return floor16(x) != floor16(y);
}

class Scene {
public:
    Scene() : perSampleUbo(256 * MAX_SAMPLE_COUNT) {}

    void Init() {
        frameTexture.Init(GL_RGBA32F, windowWidth, windowHeight, "mipmap", nullptr);
        frameSunTexture.Init(GL_RGBA32F, windowWidth, windowHeight, "mipmap", nullptr);
        bloomTexture.Init(GL_RGBA32F, windowWidth, windowHeight, "linear", nullptr);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameTexture.id, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, frameSunTexture.id, 0);
        const GLenum buffers[]{ GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glNamedFramebufferDrawBuffers(fbo, 2, buffers);

        distortionReuse.Init(GL_RGBA32F, windowWidth, windowHeight, "nearest", nullptr);

        glGenFramebuffers(1, &bloomFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, bloomFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bloomTexture.id, 0);

        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);

        perFrameCpuUbo.prevRegenCameraPosition.get() = vec3(-10000.0f);
        perSampleUbo.cameraPosition.get() = vec3(GetCameraPos(0.0 + TIME_OFFSET));

        beatsSSBO.Init(sizeof(BeatStructGPU) * beatsArray2.size());
        glNamedBufferSubData(beatsSSBO.id, 0, sizeof(BeatStructGPU) * beatsArray2.size(), beatsArray2.data());
        beatsSSBO.Bind(4);
        Dispatch(initBeatStruct, RoundUpDiv(beatsArray2.size(), 256), 1, 1);

        portalRangesSSBO.Init(256 * sizeof(WorldRange));
        glNamedBufferSubData(portalRangesSSBO.id, 0, 256 * sizeof(WorldRange), portalPositions.data());

        physicalFromLogicalSSBO.Init(1024 * sizeof(LogicalID));
        glNamedBufferSubData(physicalFromLogicalSSBO.id, 0, 1024 * sizeof(LogicalID), physicalFromLogical.data());

        terrain.Init();
    }

    void RenderFrame(float time, int frameID) {
        if (!initialized) {
            Init();
        }

        beatsSSBO.Bind(4);
        portalRangesSSBO.Bind(7);
        physicalFromLogicalSSBO.Bind(8);


        int samples = SAMPLE_COUNT;
        if (interactive &&
            glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED &&
            glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            samples = 512;
        }

        ProcessInput(movement, samples);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        distortionReuse.BindImage(3);
        frameTexture.BindImage(4);
        frameSunTexture.BindImage(5);

        terrain.Bind();

        perFrameCpuUbo.resetCamera.get() = 0;

        if (KeyPressEvent(GLFW_KEY_5) || !initialized) {
            campitch = 0.0f;
            camyaw = 0.0f;
            movement = vec3(0.0);
            zoom = 0.0;
            perFrameCpuUbo.resetCamera.get() = 1;
        }

        { // CPU Uniform data
            perFrameCpuUbo.yaw = camyaw + customyaw;
            perFrameCpuUbo.pitch = campitch + custompitch;
            perFrameCpuUbo.zoom = zoom + customzoom;
            perFrameCpuUbo.nonBlurTime.get() = time;
            perFrameCpuUbo.frameID = frameID;
            perFrameCpuUbo.shaderReload = int(terrain.shaderIncrement != ShaderIncrement);
            perFrameCpuUbo.sampleCount = samples;
            perFrameCpuUbo.currMovement = movement + custommovement;
            perFrameCpuUbo.BindAndUpload(1);
        }

        if (KeyPressEvent(GLFW_KEY_4)) {
            cout << "yaw: " << perFrameCpuUbo.yaw.get() << ", pitch: " << perFrameCpuUbo.pitch.get() << ", zoom: " << perFrameCpuUbo.zoom.get() << ", movement: (" << perFrameCpuUbo.currMovement.get().x << ", " << perFrameCpuUbo.currMovement.get().y << ", " << perFrameCpuUbo.currMovement.get().z << ")\n";
        }

        perSampleUbo.Bind(14, GL_SHADER_STORAGE_BUFFER);
        Dispatch(uniformShader, RoundUpDiv(samples, 1024), 1, 1);
        glGetNamedBufferSubData(perSampleUbo.id, 0, 256, perSampleUbo.dataCPU);
        perSampleUbo.BindRange(0, 0, 256, GL_UNIFORM_BUFFER);
        perFrameCpuUbo.prevWorldID.get() = perSampleUbo.uWorldID.get();
        perFrameCpuUbo.prevFrameCameraPosition.get() = perSampleUbo.cameraPosition.get();
        
        if (terrain.shaderIncrement != ShaderIncrement
            || diff16(perSampleUbo.cameraPosition.get().x, perFrameCpuUbo.prevRegenCameraPosition.get().x)
            || diff16(perSampleUbo.cameraPosition.get().z, perFrameCpuUbo.prevRegenCameraPosition.get().z)
            ) {
            terrain.Regenerate(terrain.shaderIncrement != ShaderIncrement);
            terrain.shaderIncrement = ShaderIncrement;
            perFrameCpuUbo.prevRegenCameraPosition.get() = perSampleUbo.cameraPosition.get();
        }

        for (int i = 0; i < samples; ++i) {
            perSampleUbo.BindRange(0, 256 * i, 256, GL_UNIFORM_BUFFER);
            Dispatch(composite0, windowWidth / 16, windowHeight / 16, 1, false);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glDisable(GL_BLEND);

        frameSunTexture.GenerateMipmap();
        frameTexture.GenerateMipmap();
        frameSunTexture.Bind(9, GL_TEXTURE_2D);
        frameTexture.Bind(13, GL_TEXTURE_2D);
        glBindFramebuffer(GL_FRAMEBUFFER, bloomFBO);
        Draw(bloom, VAO, GL_TRIANGLES, 0, 6);

        bloomTexture.Bind(10, GL_TEXTURE_2D);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        Draw(tonemap, VAO, GL_TRIANGLES, 0, 6);

        initialized = true;
    }

    unsigned int VAO;
    Terrain terrain;
    Texture frameTexture;
    Texture bloomTexture;
    Texture frameSunTexture;
    Texture distortionReuse;
    SSBO beatsSSBO;
    SSBO portalRangesSSBO;
    SSBO physicalFromLogicalSSBO;
    unsigned int fbo;
    unsigned int bloomFBO;
    PerFrameCpuUbo perFrameCpuUbo;
    PerSampleUbo perSampleUbo;

    vec3 movement = vec3(0.0);

    bool initialized = false;
};

#define WRITE_THREAD_COUNT 32

size_t frameBytes = 3 * windowWidth * windowHeight;
uint8* frame2CPU = new uint8[3 * windowWidth * windowHeight * WRITE_THREAD_COUNT];

void WriteImage(int frameID, int threadID) {
    int threadOffset = frameBytes * threadID;

    char buff[128];
    snprintf(buff, sizeof(buff), "Frames/ebin%05d.png", frameID);
    stbi_write_png(buff, windowWidth, windowHeight, 3, frame2CPU + threadOffset, windowWidth * 3);
}

static double lastX = 0.0;
static double lastY = 0.0;

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
        vec2 delta;
        delta.x = (xpos - lastX) / windowWidth;
        delta.y = float(ypos - lastY) / windowHeight;

        camVelocity += delta * vec2(camMoments[0]);
    }

    lastX = xpos;
    lastY = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    zoom -= yoffset * 0.1f;
}


void ReloadShaders(bool regenTerrain) {
    ReloadingShaders = true;
    
    for (auto i = ReloadableShaders.begin(); i != ReloadableShaders.end(); i++) {
        
        std::string source = ReadFile(std::get<0>(i->first), std::get<1>(i->first));
        int shader = CreateProgram(source, std::get<2>(i->first), std::get<0>(i->first), std::get<1>(i->first));
        if (shader != -1) {
            *(i->second) = shader;
            system("cls");
            printf("Reloaded Shaders\n");
            if (regenTerrain) ShaderIncrement++;
        }
        else {
            system("cls");
            CreateProgram(source, std::get<2>(i->first), std::get<0>(i->first), std::get<1>(i->first));
            break;
        }
    }

    ReloadingShaders = false;
}

static bool shouldClose = false;
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        shouldClose = true;
    }

    if (key == GLFW_KEY_T && action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        ReloadShaders(false);
    }

    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
        ReloadShaders(true);
    }
}

ivec2 ATLAS_SIZE = ivec2(1024, 1024);

struct BlockTexture {
    ivec2 offset;
    int index;
};

unordered_map<string, BlockTexture> textureNames = {
    {"stone.png",            {ivec2( 0, 0),  id_stone} },
    {"grass_block_side_gray.png", {ivec2(16, 0),  id_grass} },
    {"dirt.png",             {ivec2(32, 0),  id_dirt} },
    {"grass_block_top.png",  {ivec2(48, 0),  id_grass_top} },
    {"sand.png",  {ivec2(64, 0),  id_sand} },
    {"oak_log_side.png",  {ivec2(80, 0),  id_oak_log} },
    {"oak_log_top.png",  {ivec2(96, 0),  id_oak_log_top} },
    {"leaves_transparent_gray.png",  {ivec2(112, 0),  id_leaves} },
    {"cobblestone.png",            {ivec2( 0, 16),  id_permastone} },
    {"torch2.png",            {ivec2(16, 16),  id_torch} }
};

void LoadTextureIntoAtlas(vector<uint32_t>& atlasPixels, const string& textureName) {
    int width, height, channels;
    string fileName = "textures/" + textureName;
    unsigned char* data = stbi_load(fileName.c_str(), &width, &height, &channels, 0);

    if (channels == 3) {
        unsigned char* data2 = (unsigned char*)malloc(width * height * 4);
        for (int i = 0; i < width * height; ++i) {
            data2[i*4 + 0] = data[i*3 + 0];
            data2[i*4 + 1] = data[i*3 + 1];
            data2[i*4 + 2] = data[i*3 + 2];
            data2[i*4 + 3] = 255;
        }

        data = data2;
    }

    for (int x = 0; x < 16; ++x) {
        for (int y = 0; y < 16; ++y) {
            ivec2 atlasPos = textureNames[textureName].offset + ivec2(x, y);

            atlasPixels[atlasPos.x + atlasPos.y * ATLAS_SIZE.x] = ((uint32_t*)(data))[x + y * width];
        }
    }
}

int main() {
    
    beatsArray2.reserve(beatsArray.size());
    BeatStruct curr;
    portalPositions.push_back({.zStart = -10000000,.logicalWorldID=WORLD_NAME(0)});
    physicalFromLogical[WORLD_NAME(0)].id = 0;

    int currPhysicalID = 1;
    int currLogicalID = WORLD_NAME(0);
    for (int i = 0; i < beatsArray.size(); ++i) {
        BeatStruct elem = beatsArray[i];

        if (elem.b == beat_marker_start) {
            if (elem.bt != beat_type_default) curr.bt = elem.bt;
            if (elem.d != 0.0f) curr.d = elem.d;
            continue;
        } else if (elem.b == beat_marker_end) {
            curr = BeatStruct();
            continue;
        }

        if (curr.bt != beat_type_default) elem.bt = curr.bt;
        if (curr.d != 0.0f) elem.d = curr.d;

        if (elem.bt == beat_type_portal) {
            int physicalID = currPhysicalID % MAX_WORLD_COUNT;
            portalPositions.push_back({.zStart=int(GetBeatPos(elem.b)),.physicalWorldID=physicalID,.logicalWorldID=elem.targetWorldName});
            physicalFromLogical[elem.targetWorldName].id = physicalID;
            physicalFromLogical[elem.targetWorldName].prevLogical = currLogicalID;
            currLogicalID = elem.targetWorldName;
            currPhysicalID++;
        }

        elem.b += elem.d; // Merge beat delay into beat time
        beatsArray2.push_back({.beat=elem.b, .type=elem.bt, .portalTarget=elem.targetWorldName});
    }

    for (int i = 1; i < portalPositions.size(); ++i) {
        portalPositions[i-1].zEnd = portalPositions[i].zStart;
    }
    for (int i = 0; i < portalPositions.size(); ++i) {
        portalPositions[i].zStart -= WORLD_SIZE.z/2 + 32;
        portalPositions[i].zEnd += WORLD_SIZE.z/2 + 32;
    } portalPositions.back().zEnd = 100000000;

    for (int i = 0; i < physicalFromLogical.size(); ++i) {
        if (physicalFromLogical[i].prevLogical != -1) {
            physicalFromLogical[physicalFromLogical[i].prevLogical].nextLogical = i;
        }
    }


    stbi_flip_vertically_on_write(1);
    stbi_set_flip_vertically_on_load(1);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (!doubleBuffer) glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);

    window = glfwCreateWindow(windowWidth, windowHeight, "Acid Engine", NULL, NULL);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);

    if (vSync) glfwSwapInterval(1);
    else       glfwSwapInterval(0);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugOutput, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_MEDIUM, 0, nullptr, GL_TRUE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    glViewport(0, 0, windowWidth, windowHeight);

    if (writeFrames && cleanFramesFolder) {
        std::filesystem::remove_all("Frames/");
        std::filesystem::create_directories("Frames/");
    } else if (writeFrames) {
        std::filesystem::remove_all("Frames/Shaders/");
    }

    if (writeFrames) {
        std::ofstream myFile("Frames/START_FRAME.txt");
        if (myFile.is_open()) {
            myFile << (START_FRAME - 1);
            myFile.close();
        }
    }

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    fstream myFile("textures/sky.bin", std::ios::in | std::ios::binary);
    myFile.seekg(0);
    myFile.read(skyLutData, skyLutSize);
    
    Time start;
    Time print;
    int frameID = START_FRAME;

    int width, height, channels;
    unsigned char* data = stbi_load("textures/noise.png", &width, &height, &channels, 0);
    Texture texture2(GL_RGB8, width, height, "linear", data);
    texture2.Bind(0, GL_TEXTURE_2D);

    width, height, channels;
    data = stbi_load("textures/lookup2.png", &width, &height, &channels, 0);
    Texture texture3(channels == 4 ? GL_RGBA8 : GL_RGB8, width, height, "linear", data);
    texture3.Bind(3, GL_TEXTURE_2D);

    float prevTime = 0.0;
    int prevFrameID = 0;

    float totalRenderingTime = 0.0;
    float prev = print.seconds();
    

    vector<thread> threads(WRITE_THREAD_COUNT);

    Scene scene;

    ReloadShaders(true);

    SSBO atlasSSBO;
    atlasSSBO.Init(1024 * sizeof(int) * 2);
    vector<ivec2> atlasOffsets(1024, ivec2(0));
    for (auto it : textureNames) {
        atlasOffsets[it.second.index] = it.second.offset;
    }
    glNamedBufferSubData(atlasSSBO.id, 0, 1024 * sizeof(int) * 2, atlasOffsets.data());
    atlasSSBO.Bind(15);

    Texture atlasOffsetTexture;
    atlasOffsetTexture.Init(GL_RG32F, 1024, "nearest", atlasOffsets.data());
    atlasOffsetTexture.Bind(11, GL_TEXTURE_1D);

    
    vector<uint32_t> atlasPixels(ATLAS_SIZE.x * ATLAS_SIZE.y);
    for (auto it : textureNames) {
        LoadTextureIntoAtlas(atlasPixels, it.first);
    }
    Texture atlasTexture;
    atlasTexture.Init(GL_RGBA8, ATLAS_SIZE.x, ATLAS_SIZE.y, "tinyMipmap", atlasPixels.data());
    atlasTexture.Bind(2, GL_TEXTURE_2D);

    bool justRenderedGFrame = false;
    while (!shouldClose) {
        //glfwWaitEvents();
        glfwPollEvents();

        if (interactive && glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(int(5)));
            //frameID--;
        }

        if (vSync) {
            if (print.seconds() - prev < 1.0 / framerate) {
                continue;
            } else {
                prev = print.seconds();
            }
        }
        
        float frameStartTime = start.seconds();
        float time = (interactive?START_FRAME:frameID) / framerate;

        if (frameID == START_FRAME
         || !interactive
         || glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED
         || (!justRenderedGFrame && glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)) {
            justRenderedGFrame = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;

            scene.RenderFrame(time, frameID);
        

        if (writeFrames) {
            if (frameID == START_FRAME) {
                std::filesystem::create_directories("Frames/Shaders/");
                std::filesystem::copy("Shaders/", "Frames/Shaders/");
            }
            
            int threadID = frameID % threads.size();
            int threadOffset = frameBytes * threadID;

            auto& currThread = threads[threadID];
            if (currThread.joinable()) currThread.join();

            glBindFramebuffer(GL_FRAMEBUFFER, scene.fbo);
            glReadnPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, frameBytes, frame2CPU + threadOffset);

            currThread = thread(WriteImage, frameID - (cleanFramesFolder ? START_FRAME - 1 : 0), threadID);
        }

        brBlitNamedFramebuffer(scene.fbo, 0, windowWidth, windowHeight);

        if (doubleBuffer) glfwSwapBuffers(window);
        else              glFlush();
        }

        if (KeyPressEvent(GLFW_KEY_F2)) {
            glBindFramebuffer(GL_FRAMEBUFFER, scene.fbo);
            glReadnPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, frameBytes, frame2CPU);

            const auto now = std::chrono::system_clock::now();
            std::string filename = std::format("{:%Y-%m-%d_%H_%M_%OS}_{:1}", now, (std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()%1000)/100);
            std::cout << filename << "\n";

            std::string path = std::format("Screenshots/{}.png", filename);
            stbi_write_png(path.c_str(), windowWidth, windowHeight, 3, frame2CPU + 0, windowWidth * 3);

            std::string folder = std::format("Screenshots/{}", filename);
            std::filesystem::create_directories(folder);

            std::filesystem::copy("Shaders/", folder);
        }

        //if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            camVelocity = mix(camVelocity, vec2(0.0), camMoments[3]);
            posVelocity = mix(posVelocity, vec3(0.0), displaceMoments[3]);
        //}

        if (print.seconds() >= 1.0f) {
            float s = start.seconds();
            //float fps = frameID - prevFrameID + 1;
            float fps = (frameID - prevFrameID) / totalRenderingTime;
            prevFrameID = frameID;
            totalRenderingTime = 0.0;

            print.reset();
            prev = print.seconds();

            char buff[128];
            snprintf(buff, sizeof(buff), "Acid Engine: %.0ffips, frame:%d, %2.1fms, %.1f%%", fps, frameID, 1000.0f/fps, ((frameID - START_FRAME) / framerate) / maxSeconds * 100.0);
            glfwSetWindowTitle(window, buff);
        }

        if (writeFrames && maxSeconds > 0.0 && (frameID - START_FRAME) / framerate > maxSeconds) break;

        frameID++;
        totalRenderingTime += start.seconds() - frameStartTime;
    }

    for (int i = 0; i < threads.size(); ++i) { if (threads[i].joinable()) { threads[i].join(); } }

    if (encodeVideo) {
        std::system("cd VegasClips && powershell ./CreateClip.ps1");
    }
}
