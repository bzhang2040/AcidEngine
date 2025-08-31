#include "glad.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <map>

#define CXX_STAGE
#include "Camera.glsl"
#include "Scene.glsl"
#include "Include.glsl"
#include "PrecomputeSky.glsl"
#include "sky.glsl"
#include "Uniforms.glsl"
#include "FullscreenBlit.glsl"
#include "Triangulate.glsl"
#include "Render.glsl"
#include "Bloom.glsl"
#include "Tonemap.glsl"

#include <fstream>

class Time {
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    Time() {
        start = std::chrono::high_resolution_clock::now();
    }

    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }

    float seconds() {
        return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count()) / 1000.0f;
    }
};

#define brCopyImageSubData(source, dest, width, height) glCopyImageSubData(source, GL_TEXTURE_2D, 0, 0 ,0, 0, dest, GL_TEXTURE_2D, 0, 0, 0, 0, width, height, 1)
#define brBlitNamedFramebuffer(source, dest, width, height) glBlitNamedFramebuffer(source, dest, 0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
#define brClearColor(color) glClearColor(color.r, color.g, color.b, color.a); glClear(GL_COLOR_BUFFER_BIT);

void APIENTRY glDebugOutput(GLenum source,
    GLenum type,
    unsigned int id,
    GLenum severity,
    GLsizei length,
    const char* message,
    const void* userParam)
{
    // ignore non-significant error/warning codes
    //if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 

    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " << message << std::endl;

    switch (source) {
    case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
    case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
    case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
    } std::cout << std::endl;

    switch (type) {
    case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
    case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
    case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
    case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
    case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
    case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
    } std::cout << std::endl;

    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
    case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
    case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
    case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
    } std::cout << std::endl;
    std::cout << std::endl;
}

bool CheckCompileErrors(GLuint shader, std::string type)
{
    GLint success;
    GLchar infoLog[1024];
    if (type == "shader") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            return false;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            return false;
        }
    }

    return true;
}

void PrintStringLineNumbers(const std::string& str) {
    
    std::istringstream iss(str);

    int lineNum = 1;
    for (std::string line; std::getline(iss, line); ) {
        std::cout << lineNum++ << ": " << line << "\n";
    }
}

class UBOEntryBase {
public:
    //UBOEntryBase(int floatOffset_) : floatOffset(floatOffset_), parentData(nullptr) {}

    int floatOffset;
    int* parentData;

    virtual int FloatAlignment() = 0;
    virtual int SizeFloats() = 0;
};

template <typename T>
class UBOEntry : public UBOEntryBase {
public:
    //UBOEntry(int floatOffset_) : UBOEntryBase(floatOffset_) {}

    void operator=(const T& other) {
        *((T*)(&parentData[floatOffset])) = other;
    }

    T& get() {
        return *((T*)(&parentData[floatOffset]));
    }

    int FloatAlignment() override {
        if (std::is_same_v<T, float>) return 1;
        if (std::is_same_v<T, int>) return 1;
        if (std::is_same_v<T, glm::vec2>) return 2;
        if (std::is_same_v<T, glm::ivec2>) return 2;
        if (std::is_same_v<T, glm::uvec2>) return 2;
        if (std::is_same_v<T, glm::vec3>) return 4;
        if (std::is_same_v<T, glm::ivec3>) return 4;
        if (std::is_same_v<T, glm::uvec3>) return 4;
        if (std::is_same_v<T, glm::vec4>) return 4;
        if (std::is_same_v<T, glm::ivec4>) return 4;
        if (std::is_same_v<T, glm::uvec4>) return 4;
        if (std::is_same_v<T, glm::mat4>) return 16;
        return alignof(T) / 4;
    }

    int SizeFloats() override {
        return sizeof(T) / 4;
    }
};

int RoundUp(int numToRound, int multiple) {
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

int RoundUpDiv(int numToRound, int multiple) {
    return (numToRound + multiple - 1) / multiple;
}

class UBOBase {
public:
    std::vector<UBOEntryBase*> members;

    GLuint id;
    int size;
    int dataCPU[1024];

    UBOBase(int size_ = 1024) {
        size = size_;

        glGenBuffers(1, &id);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, id);
        glBufferData(GL_UNIFORM_BUFFER, size * sizeof(float), nullptr, GL_STATIC_DRAW);
    }

    void DoAlignment() {
        int offset = 0;
        for (int i = 0; i < members.size(); ++i) {
            members[i]->parentData = dataCPU;
            members[i]->floatOffset = RoundUp(offset, members[i]->FloatAlignment());
            offset = members[i]->floatOffset + members[i]->SizeFloats();
        }
    }

    void Bind(int binding, int target) {
        glBindBufferBase(target, binding, id);
    }

    void BindRange(int binding, int offset, int size, int target) {
        glBindBufferRange(target, binding, id, offset, size);
    }

    void BindAndUpload(int binding) {
        glBindBufferBase(GL_UNIFORM_BUFFER, binding, id);
        glBufferData(GL_UNIFORM_BUFFER, size * sizeof(float), dataCPU, GL_STATIC_DRAW);
    }
};

class PerFrameCpuUbo : public UBOBase {
public:
    PER_FRAME_CPU_UBO(UBO_DEF);

    PerFrameCpuUbo(int size_ = 1024) : UBOBase(size_) {
        PER_FRAME_CPU_UBO(UBO_PUSH);
        DoAlignment();
    }
};

class PerSampleUbo : public UBOBase {
public:
    PER_SAMPLE_UBO(UBO_DEF);

    PerSampleUbo(int size_ = 1024) : UBOBase(size_) {
        PER_SAMPLE_UBO(UBO_PUSH);
        DoAlignment();
    }
};

std::string ReadFile(std::string filename, std::string includeName = "") {
    std::string dirPath = "Shaders/";
    std::string fullPath = dirPath + filename;

    std::ifstream t(fullPath);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string ebin2 = buffer.str();

    if (includeName == "") {
        for (int i = 0; i < filename.size(); ++i) { if (filename[i] == '.') { filename[i] = '_'; }; filename[i] = std::toupper(filename[i]); };
        includeName = filename;
    }
    ebin2 = //"#define " + includeName + "\n" +
        ebin2;
    return ebin2;
}

static bool ReloadingShaders = false;
static int ShaderIncrement = 0;

std::string Replace(const std::string& str, const std::string& from, const std::string& to) {
    std::string newString = str;
    size_t start_pos = newString.find(from);
    if (start_pos == std::string::npos) {
        return str;
    }
    newString.replace(start_pos, from.length(), to);
    return newString;
}

std::string AddPreamble(const std::string& type_, std::string sourceText, const std::string& filename, const std::string& preprocessorName) {
    std::string modifiedText;

    modifiedText += ReadFile("Camera.glsl");
    modifiedText += ReadFile("Scene.glsl");

    modifiedText = Replace(modifiedText, "//#include INCLUDE\n", ReadFile("Include.glsl"));

    std::string target = "//#include SKY\n";
    size_t targetPos = sourceText.find(target);
    if (targetPos != string::npos) {
        sourceText.replace(targetPos, target.length(), ReadFile(PrecomputeSky_glsl) + ReadFile(Sky_glsl));
    }

    modifiedText += sourceText;
    if (type_ == "vertex")   modifiedText = "#define VERTEX_STAGE\n" + modifiedText;
    else if (type_ == "fragment") modifiedText = "#define FRAGMENT_STAGE\n" + modifiedText;
    else if (type_ == "compute")  modifiedText = "#define COMPUTE_STAGE\n" + modifiedText;

    modifiedText = std::string("#define BEATS_COUNT ") + std::to_string(beatsArray2.size()) + std::string("\n") + modifiedText;

    modifiedText = "#define " + preprocessorName + "\n" + modifiedText;

    if (filename != "Render.glsl") {
        modifiedText = R"===(
#extension GL_NV_shader_thread_group : enable
#extension GL_NV_shader_thread_shuffle : enable
)===" + modifiedText;
    }

    modifiedText = R"===(#version 450
#extension GL_NV_gpu_shader5 : enable
#pragma optionNV(fastmath on)
#pragma optimize(on)
#extension GL_ARB_shader_ballot : enable
)===" + modifiedText;

    return modifiedText;
}

int CreateShader(const std::string& type_, std::string sourceText, const std::string& filename, const std::string& preprocessorName) {
    std::string modifiedText = AddPreamble(type_, sourceText, filename, preprocessorName);
    
    int type;
    if (type_ == "vertex") type = GL_VERTEX_SHADER;
    else if (type_ == "fragment") type = GL_FRAGMENT_SHADER;
    else if (type_ == "compute") type = GL_COMPUTE_SHADER;
    else assert(false);
    int shader = glCreateShader(type);

    const char* cstr = modifiedText.c_str();

    glShaderSource(shader, 1, &cstr, NULL);
    glCompileShader(shader);

    if (!CheckCompileErrors(shader, "shader")) {
        std::cout << "FAILED TO COMPILE SHADER\n";
        PrintStringLineNumbers(modifiedText);
        if (!ReloadingShaders) throw std::runtime_error("");
        return -1;
    }

    return shader;
}

int CreateGraphicsProgram(const std::string& text, const std::string& filename, const std::string& preprocessorName) {
    int vShader = CreateShader("vertex", text, filename, preprocessorName);
    int fShader = CreateShader("fragment", text, filename, preprocessorName);

    if (ReloadingShaders && (vShader == -1 || fShader == -1)) {
        return -1;
    }

    unsigned int program = glCreateProgram();
    glAttachShader(program, vShader);
    glAttachShader(program, fShader);

    glLinkProgram(program);

    if (!CheckCompileErrors(program, "program")) {
        std::cout << "FAILED TO LINK PROGRAM\n";
        if (!ReloadingShaders) throw std::runtime_error("");
        return -1;
    }

    return program;
}

int CreateProgram(const std::string& text, const std::string& type, const std::string& filename, const std::string& preprocessorName) {
    if (type == "graphics") {
        return CreateGraphicsProgram(text, filename, preprocessorName);
    }

    int cShader = CreateShader("compute", text, filename, preprocessorName);

    unsigned int program = glCreateProgram();
    glAttachShader(program, cShader);

    glLinkProgram(program);

    if (!CheckCompileErrors(program, "program")) {
        std::cout << "FAILED TO LINK PROGRAM\n";
        if (!ReloadingShaders) throw std::runtime_error("");
        return -1;
    }

    return program;
}

// <Filename, PreprocessorName, ShaderType, ShaderSource>
std::map<std::tuple<std::string, std::string, std::string>, int*> ReloadableShaders;

class Shader {
public:
    int id = 0;
    std::string type;

    Shader() {}

    Shader(const Shader& other) {
        id = other.id;
        type = other.type;
    }

    Shader operator=(Shader& other) {
        id = other.id;
        type = other.type;

        return *this;
    }

    Shader operator=(const Shader& other) {
        id = other.id;
        type = other.type;

        return *this;
    }

    Shader(const std::string& text, const std::string& type_) {
        type = type_;
        id = CreateProgram(text, type, "", "");
    }

    Shader(const std::string& filename, const std::string prename, const std::string& type_) {
        type = type_;
        std::string source = ReadFile(filename, prename);
        //id = CreateProgram(source, type);
        ReloadableShaders[std::make_tuple(filename, prename, type)] = &id;
    }
};

void Dispatch(const Shader& shader, int x, int y, int z, bool barrier = true) {
    if (shader.type != "compute") {
        throw std::runtime_error("");
    }

    glUseProgram(shader.id);
    glDispatchCompute(x, y, z);
    if (barrier) glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void DispatchIndirect(const Shader& shader, int offset, bool barrier = true) {
    if (shader.type != "compute") {
        throw std::runtime_error("");
    }

    glUseProgram(shader.id);
    glDispatchComputeIndirect(offset);
    if (barrier) glMemoryBarrier(GL_ALL_BARRIER_BITS);
}


void Draw(const Shader& shader, int VAO, int topology, int offset, int count) {
    if (shader.type != "graphics") {
        throw std::runtime_error("");
    }

    glBindVertexArray(VAO);
    glUseProgram(shader.id);
    glDrawArrays(topology, offset, count);
}

void DrawIndirect(const Shader& shader, int VAO, int topology, int offset) {
    if (shader.type != "graphics") {
        throw std::runtime_error("");
    }

    glBindVertexArray(VAO);
    glUseProgram(shader.id);
    glDrawArraysIndirect(topology, (const void*)offset);
}

void DrawIndirectCount(const Shader& shader, int VAO, int topology) {
    if (shader.type != "graphics") {
        throw std::runtime_error("");
    }

    glBindVertexArray(VAO);
    glUseProgram(shader.id);
    //(GLenum mode, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride);
    glMultiDrawArraysIndirectCount(topology, (const void*)16, 0, 4, 16);
}

class Texture {
public:
    GLuint id;
    int width;
    int height;
    int depth;
    int dim;
    GLint format;
    std::string filtering;

    Texture() {}

    Texture(GLint format_, int width_, int height_, std::string filtering_, void* data)
    : width(width_), height(height_), format(format_), filtering(filtering_), dim(2) {
        Init(format_, width_, height_, filtering_, data);
    }

    Texture(GLint format_, int width_, int height_, int depth_, std::string filtering_, void* data) {
        Init(format_, width_, height_, depth_, filtering_, data);
    }

    void Init(GLint format_, int width_, std::string filtering_, void* data) {
        width = width_; format = format_; filtering = filtering_; dim = 1;

        glGenTextures(1, &id);
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(GL_TEXTURE_1D, id);

        GLint form, type;
        GetFormats(format, form, type);
        glTexImage1D(GL_TEXTURE_1D, 0, format, width, 0, form, type, data);

        if (filtering == "mipmap" || filtering == "nearestMipmap" || filtering == "tinyMipmap") {
            //assert(data);
            glGenerateTextureMipmap(id);
        }

        SetFiltering();
    }

    void Init(GLint format_, int width_, int height_, std::string filtering_, void* data) {
        width = width_; height = height_; format = format_; filtering = filtering_; dim = 2;
        
        glGenTextures(1, &id);
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(GL_TEXTURE_2D, id);

        GLint form, type;
        GetFormats(format, form, type);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, form, type, data);

        if (filtering == "mipmap" || filtering == "nearestMipmap" || filtering == "tinyMipmap") {
            //assert(data);
            glGenerateTextureMipmap(id);
        }

        SetFiltering();
    }

    void Init(GLint format_, int width_, int height_, int depth_, std::string filtering_, void* data) {
        width = width_; height = height_; depth = depth_; format = format_; filtering = filtering_; dim = 3;

        glGenTextures(1, &id);
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(GL_TEXTURE_3D, id);

        GLint form, type;
        GetFormats(format, form, type);
        glTexImage3D(GL_TEXTURE_3D, 0, format, width, height, depth, 0, form, type, data);

        if (filtering == "mipmap" || filtering == "nearestMipmap" || filtering == "tinyMipmap") {
            //assert(data);
            glGenerateTextureMipmap(id);
        }

        SetFiltering();
    }

    void SetFiltering() {
        if (filtering == "nearest") {
            glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        } else if (filtering == "nearestMipmap") {
            glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
            glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        } else if (filtering == "linear") {
            glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        } else if (filtering == "mipmap") {
            glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        } else if (filtering == "tinyMipmap") {
            glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
            glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        } else assert(false);
    }

    void GetFormats(GLint internalformat, GLint& format, GLint& type) {
        switch (internalformat) {
        case GL_RGBA32F: format = GL_RGBA; type = GL_FLOAT; return;
        case GL_RGBA16F: format = GL_RGBA; type = GL_HALF_FLOAT; return;
        case GL_RGBA32UI: format = GL_RGBA_INTEGER; type = GL_UNSIGNED_INT; return;
        case GL_RGBA16UI: format = GL_RGBA_INTEGER; type = GL_UNSIGNED_SHORT; return;
        case GL_RGBA16I: format = GL_RGBA_INTEGER; type = GL_SHORT; return;
        case GL_RG32F: format = GL_RG; type = GL_FLOAT; return;
        case GL_R32UI: format = GL_RED_INTEGER; type = GL_UNSIGNED_INT; return;
        case GL_R8UI: format = GL_RED_INTEGER; type = GL_UNSIGNED_BYTE; return;
        case GL_R32F: format = GL_RED; type = GL_FLOAT; return;
        case GL_DEPTH_COMPONENT: format = GL_DEPTH_COMPONENT; type = GL_FLOAT; return;
        case GL_RGBA8: format = GL_RGBA; type = GL_UNSIGNED_BYTE; return;
        case GL_RGB8: format = GL_RGB; type = GL_UNSIGNED_BYTE; return;
        }
    }

    void Bind(int slot, int sampler) {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(sampler, id);
    }

    void BindImage(int slot) {
        glBindImageTexture(slot, id, 0, GL_TRUE, 0, GL_READ_WRITE, format);
    }

    void GenerateMipmap() {
        if (filtering == "mipmap" || filtering == "nearestMipmap") {
            glGenerateTextureMipmap(id);
        }
    }

};