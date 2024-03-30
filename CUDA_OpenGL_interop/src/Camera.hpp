#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <math.h>

#include "FileLoaders.hpp"
#include "Shader.hpp"
#include "FrameHandler.hpp"

class OrthographicCameraProperties {
public:
    float min_zoom;
    float max_zoom;
    float far_speed;
    float close_speed;
};

class OrthographicCamera {
public:
    OrthographicCamera(float_t left = 0, float_t right = 1.0f, float_t bottom = 0.0f, float_t top = 1.0f, float_t near = 1.0f, float_t far = 10.0f,
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f));

    void load(const std::string& project_path, Properties& properties);
    void init(float_t left, float_t right, float_t bottom, float_t top, float_t near, float_t far,
        glm::vec3 position);

    const glm::mat4& getProjection_matrix() const { return this->projection_matrix; }
    const glm::mat4& getView_matrix() const { return this->view_matrix; }
    const glm::mat4& getProjectionViewMatrix() const { return this->projection_view_matrix; }

    void setPosition(glm::vec3 &position);
    const glm::vec3& getCameraPosition() const { return this->position; }

    void setFrustum(glm::vec4& frustum);
    const glm::vec4& getFrustum() const { return this->frustum; }

    const float getFrustumSize() const { return this->frustum_size; }
    const float getFrustumRatio() const { return this->frustum_ratio; }

    void zoomIn(FrameHandler& input);
    void zoomOut(FrameHandler& input);
    void updatePosition(FrameHandler& input);
private:

    void recalculateViewMatrix();
    void recalculateProjectionMatrix();
    void recalculateProjectionViewMatrix();

    OrthographicCameraProperties properties;
    glm::mat4 projection_matrix;
    glm::mat4 view_matrix;
    glm::mat4 projection_view_matrix;
    glm::vec3 position;
    glm::vec4 frustum;
    float frustum_ratio;
    float near_plane;
    float far_plane;
    float zoom;
    float frustum_size;
    float default_frustum_size;
};
