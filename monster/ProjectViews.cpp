#include "ProjectViews.h"
#include "MeshView.h"
#include "MapsData.h"
#include "ImageTransform.h"
#include <opencv4/opencv2/opencv.hpp>

#include <iostream>

using namespace std;
using namespace Monster;
using namespace Eigen;

bool ProjectViews::loadMaps(const MatI &masks, const MatD &depths, const Mat3D &normals)
{
    mHeight = masks.size();
    mWidth = masks[0].size();

    MatI pointIdx(mHeight, vector<int>(mWidth, -1));
    mMesh.positions.clear();
    mMesh.normals.clear();

    for (int i = 0; i < mHeight; i++)
    {
        for (int j = 0; j < mWidth; j++)
        {
            if (!masks[i][j])
                continue;
            vec3d p((double)j, (double)i, depths[i][j]);
            vec3d n = normals[i][j];
            pointIdx[i][j] = (int)mMesh.positions.size();
            mMesh.positions.push_back(p);
            mMesh.normals.push_back(n);
        }
    }
    mMesh.amount = (int)mMesh.positions.size();

    mMesh.indices.clear();
    for (int row = 0; row < mHeight - 1; row++)
    {
        for (int col = 0; col < mWidth - 1; col++)
        {
            vector<int> idx(0);
            if (pointIdx[row][col] >= 0)
                idx.push_back(pointIdx[row][col]);
            if (pointIdx[row + 1][col] >= 0)
                idx.push_back(pointIdx[row + 1][col]);
            if (pointIdx[row + 1][col + 1] >= 0)
                idx.push_back(pointIdx[row + 1][col + 1]);
            if (pointIdx[row][col + 1] >= 0)
                idx.push_back(pointIdx[row][col + 1]);
            if ((int)idx.size() >= 3)
            {
                mMesh.indices.push_back(vec3i(idx[0], idx[1], idx[2]));
            }
            if ((int)idx.size() == 4)
            {
                mMesh.indices.push_back(vec3i(idx[0], idx[2], idx[3]));
            }
        }
    }

    loadMesh(&mMesh.positions, &mMesh.normals, &mMesh.indices);

    Eigen::Matrix4d imgMat;
    MeshView::buildImageMatrix(imgMat, mWidth, mHeight);
    mInvImageMat = imgMat.inverse();
}

bool ProjectViews::project(const Eigen::Matrix4d &viewMat,
                           const Eigen::Matrix4d &rotMat,
                           MatI &masks,
                           MatD &depths,
                           Mat3D &normals)
{
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(3, vbo);
    glCheckError();

    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, V_vbo.size() * sizeof(float), V_vbo.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glCheckError();

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, V_normals_vbo.size() * sizeof(float), V_normals_vbo.data(), GL_STATIC_DRAW);
    glCheckError();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glCheckError();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, F_vbo.size() * sizeof(unsigned int), F_vbo.data(), GL_STATIC_DRAW);
    glCheckError();

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    shader.use();
    glCheckError();

    glBindFragDataLocation(shader.ID, 0, "oColor");
    glCheckError();

    Eigen::Matrix4f uViewMat = (mInvImageMat * viewMat).cast<float>();
    Eigen::Matrix4f uRotMat = rotMat.cast<float>();

    shader.setMat4("uViewMatrix", uViewMat);
    shader.setMat4("uRotateMatrix", uRotMat);
    glCheckError();

    glDrawElements(GL_TRIANGLES, F_vbo.size(), GL_UNSIGNED_INT, 0);
    glCheckError();

    vector<unsigned short> imgBuffer(mHeight * mWidth * 4);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_SHORT, &imgBuffer.front());
    glCheckError();

    ImageTransform::flipVertical(imgBuffer, imgBuffer, mWidth, mHeight, 4);

    masks.assign(mHeight, vector<int>(mWidth, false));
    depths.assign(mHeight, vector<double>(mWidth, 1.0));
    normals.assign(mHeight, vector<vec3d>(mWidth, vec3d(1.0, 1.0, 1.0)));

    int offset = 0;
    for (int row = 0; row < mHeight; row++)
    {
        for (int col = 0; col < mWidth; col++)
        {
            unsigned short r = imgBuffer[offset];
            unsigned short g = imgBuffer[offset + 1];
            unsigned short b = imgBuffer[offset + 2];
            unsigned short a = imgBuffer[offset + 3];
            double x = r / (double)32768 - 1.0;
            double y = g / (double)32768 - 1.0;
            double z = b / (double)32768 - 1.0;
            double d = a / (double)32768 - 1.0;
            vec3d n(x, y, z);
            if (d < 0.9)
            {
                masks[row][col] = true;
                depths[row][col] = d;
                normals[row][col] = n;
            }
            offset += 4;
        }
    }

    return true;
}

bool ProjectViews::project()
{
    Eigen::Matrix4d viewMat, projMat, imgMat, rotMat;
    Eigen::Vector3d viewPoint(0, 0, 2.5);
    MeshView::buildViewMatrix(viewPoint, viewMat);
    cout << "viewMat\n"
         << viewMat << endl;
    MeshView::buildProjMatrix(projMat);

    MeshView::buildImageMatrix(imgMat, 256, 256);
    auto uViewMat = (imgMat * projMat * viewMat);

    cv::Mat img = cv::Mat::zeros(256, 256, CV_8UC1);
    for (int i = 0; i < V_vbo.rows(); i++)
    {
        const Eigen::Vector3d &v = V_vbo.row(i).cast<double>();
        auto p = uViewMat * Eigen::Vector4d(v.x(), v.y(), v.z(), 1.0);
        cout << "orig: " << v.transpose() << ", p: " << p.transpose() << endl;
        int x = p.x();
        int y = p.y();
        img.at<uchar>(x, y) = 255;
    }

    cv::imwrite("./data/proj.jpg", img);
}

bool ProjectViews::init(int imgSize)
{
    mHeight = imgSize;
    mWidth = imgSize;
    if (!initContext())
        return false;
    if (!initGL(imgSize)) // Configure global opengl context
        return false;
    shader.init("resources/sketch.vs", "resources/sketch.fs");
    // shader.init("resources/framebuffers.vs", "resources/framebuffers.fs");
    // cout << __FILE__ << " " << __LINE__ << " init GL successfully" << endl;
    return true;
}

bool ProjectViews::initContext()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    window = glfwCreateWindow(mWidth, mHeight, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    // glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // glfwSetCursorPosCallback(window, mouse_callback);
    // glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    return true;
}

void ProjectViews::cleanContext()
{
    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteTextures(1, &mTexID);
    glDeleteRenderbuffers(1, &mRBID);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &mFBOID);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(3, vbo);
    glDeleteProgram(shader.ID);
    glfwTerminate();
}

bool ProjectViews::initGL(int imgSize)
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL); //when the depth is less or equal than the value in buffer, pass the test
    glDisable(GL_ALPHA_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_SMOOTH);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);
    glDisable(GL_DITHER);
    glEnable(GL_LIGHTING);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    glEnable(GL_LIGHT0);

    glGenTextures(1, &mTexID);
    glBindTexture(GL_TEXTURE_2D, mTexID);
    // // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imgSize, imgSize, 0, GL_RGBA, GL_UNSIGNED_SHORT, NULL);

    // // Frame Buffer Object
    glGenFramebuffers(1, &mFBOID);
    glBindFramebuffer(GL_FRAMEBUFFER, mFBOID);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTexID, 0);
    // // Render Buffer Object
    glGenRenderbuffers(1, &mRBID);
    glBindRenderbuffer(GL_RENDERBUFFER, mRBID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, imgSize, imgSize);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mRBID);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        cerr << "Frame buffer error\n";
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, (GLsizei)imgSize, (GLsizei)imgSize);
    return true;
}

void ProjectViews::render()
{
    //TODO: fix the bug that the image does not show
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // render
        // ------
        // bind to framebuffer and draw scene as we normally would to color texture
        glBindFramebuffer(GL_FRAMEBUFFER, mFBOID);
        glEnable(GL_DEPTH_TEST);

        shader.use();

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, F_vbo.size(), GL_UNSIGNED_INT, 0);
        // glCheckError();

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void ProjectViews::loadMesh(const vector<vec3> *vertices, const vector<vec3> *normals, const vector<vec3i> *indices)
{
    Eigen::MatrixXd V, V_Normals;
    Eigen::MatrixXi F;
    V.resize(vertices->size(), 3);
    for (int i = 0; i < vertices->size(); i++)
    {
        const vec3d &pos = (*vertices)[i];
        for (int j = 0; j < 3; j++)
        {
            V(i, j) = pos[j];
        }
    }

    if (normals)
    {
        V_Normals.resize(normals->size(), 3);
        for (int i = 0; i < normals->size(); i++)
        {
            const vec3d &n = (*normals)[i];
            for (int j = 0; j < 3; j++)
            {
                V_Normals(i, j) = n[j];
            }
        }
    }

    if (indices)
    {
        F.resize(indices->size(), 3);
        for (int i = 0; i < indices->size(); i++)
        {
            const vec3i &f = (*indices)[i];
            for (int j = 0; j < 3; j++)
            {
                F(i, j) = f[j];
            }
        }
    }

    V_vbo = V.cast<float>();
    V_normals_vbo = V_Normals.cast<float>();
    F_vbo = F.cast<unsigned int>();
    cout << __FILE__ << " " << __LINE__ << " load mesh successfully\n";
    cout << "# vertices=" << V_vbo.rows() << endl;
    cout << "# normals=" << V_normals_vbo.rows() << endl;
    cout << "# faces=" << F_vbo.rows() << endl;
}

void ProjectViews::processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

void ProjectViews::bindBuffers()
{
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(3, vbo);
    glCheckError();

    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, V_vbo.size() * sizeof(float), V_vbo.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glCheckError();

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, V_normals_vbo.size() * sizeof(float), V_normals_vbo.data(), GL_STATIC_DRAW);
    glCheckError();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glCheckError();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, F_vbo.size() * sizeof(unsigned int), F_vbo.data(), GL_STATIC_DRAW);
    glCheckError();

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    shader.use();
    glCheckError();

    glBindFragDataLocation(shader.ID, 0, "oColor");
    glCheckError();

    // Eigen::Matrix4f uViewMat = (mInvImageMat * viewMat).cast<float>();
    // Eigen::Matrix4f uRotMat = rotMat.cast<float>();
    Eigen::Matrix4d viewMat, projMat, imgMat, rotMat;
    Eigen::Vector3d viewPoint(0, 0, 1.5);
    MeshView::buildViewMatrix(viewPoint, viewMat);
    MeshView::buildProjMatrix(projMat, -1.5, 1.5, -1.5, 1.5, 0, 5);

    Eigen::Matrix<float, 4, 4> uViewMat = (projMat * viewMat).cast<float>();
    rotMat.setIdentity();
    rotMat.topLeftCorner(3, 3) = viewMat.topLeftCorner(3, 3);
    Eigen::Matrix<float, 4, 4> uRotMat = rotMat.cast<float>();
    cout << "viewMat\n"
         << viewMat << endl;

    shader.setMat4("uViewMatrix", uViewMat);
    shader.setMat4("uRotateMatrix", uRotMat);
    glCheckError();

    glDrawElements(GL_TRIANGLES, F_vbo.size(), GL_UNSIGNED_INT, 0);
    glCheckError();

    vector<unsigned short> imgBuffer(mHeight * mWidth * 4);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_SHORT, &imgBuffer.front());
    float depth;
    // Qt uses upper corner for its origin while GL uses the lower corner.
    glReadPixels(mWidth / 2, mHeight / 2, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    cout << __FILE__ << " " << __LINE__ << " center depth=" << depth << endl;
    glCheckError();

    ImageTransform::flipVertical(imgBuffer, imgBuffer, mWidth, mHeight, 4);

    MatI masks;
    MatD depths;
    Mat3D normals;
    masks.assign(mHeight, vector<int>(mWidth, false));
    depths.assign(mHeight, vector<double>(mWidth, 1.0));
    normals.assign(mHeight, vector<vec3d>(mWidth, vec3d(1.0, 1.0, 1.0)));

    int offset = 0;
    for (int row = 0; row < mHeight; row++)
    {
        for (int col = 0; col < mWidth; col++)
        {
            unsigned short r = imgBuffer[offset];
            unsigned short g = imgBuffer[offset + 1];
            unsigned short b = imgBuffer[offset + 2];
            unsigned short a = imgBuffer[offset + 3];
            double x = r / (double)32768 - 1.0;
            double y = g / (double)32768 - 1.0;
            double z = b / (double)32768 - 1.0;
            double d = a / (double)32768 - 1.0;
            vec3d n(x, y, z);
            if (d < 0.9)
            {
                masks[row][col] = true;
                depths[row][col] = d;
                normals[row][col] = n;
            }
            offset += 4;
        }
    }

    if (true)
    {
        MapsData::visualizeMask("./data/mask.jpg", masks);
        MapsData::visualizeDepth("./data/depth.jpg", depths);
        MapsData::visualizeNormal("./data/normal.jpg", normals);
    }

    // glBindVertexArray(0);
    // glDeleteBuffers(3, vbo);
    // glDeleteVertexArrays(1, &vao);
}