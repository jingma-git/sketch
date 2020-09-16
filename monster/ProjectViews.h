#pragma once

#include <vector>
#include <string>
#include <eigen3/Eigen/Eigen>
#include "Shader.h"
#include "Types.h"

#include "glad/glad.h" //glad must be put before GLFW
#include "GLFW/glfw3.h"

using namespace std;

inline GLenum glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
        case GL_INVALID_ENUM:
            error = "INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            error = "INVALID_VALUE";
            break;
        case GL_INVALID_OPERATION:
            error = "INVALID_OPERATION";
            break;
        case GL_STACK_OVERFLOW:
            error = "STACK_OVERFLOW";
            break;
        case GL_STACK_UNDERFLOW:
            error = "STACK_UNDERFLOW";
            break;
        case GL_OUT_OF_MEMORY:
            error = "OUT_OF_MEMORY";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            error = "INVALID_FRAMEBUFFER_OPERATION";
            break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__)

namespace Monster
{
    class ProjectViews
    {
    public:
        // Temporary copy of the content of each VBO
        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;
        RowMatrixXf V_vbo;
        RowMatrixXf V_normals_vbo;
        Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F_vbo;

    public:
        bool init(int imageSize);
        void bindBuffers();
        void render();
        void cleanContext();
        void loadMesh(const vector<vec3> *vertices, const vector<vec3> *normals = 0, const vector<vec3i> *indices = 0);

        ProjectViews() : mHeight(256), mWidth(256) {}
        ~ProjectViews() {}

    public:
        bool loadMaps(const MatI &masks, const MatD &depths, const Mat3D &normals);
        bool project(const Eigen::Matrix4d &viewMat,
                     const Eigen::Matrix4d &rotMat,
                     MatI &masks,
                     MatD &depths,
                     Mat3D &normals);
        bool project();

    private:
        bool initContext();
        bool initGL(int imgSize);
        void processInput(GLFWwindow *window);

        GLFWwindow *window;

        Shader shader;
        unsigned int mTexID;
        unsigned int mFBOID;
        unsigned int mRBID;
        GLuint vao;    // vertex array object
        GLuint vbo[3]; // vertex buffer object (position / normal / index)

        int mHeight;
        int mWidth;

        Eigen::Matrix4d mInvImageMat;

        TTriangleMesh mMesh;
    };
} // namespace Monster