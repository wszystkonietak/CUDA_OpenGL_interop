#include "Canvas.cuh"

void Canvas::init(glm::vec4&& boundings, glm::vec2&& texture_size, std::string&& shaders_path)
{
    this->boundings = std::move(boundings);
    this->texture_size = std::move(texture_size);
    shader = Shader(shaders_path + "/canvas.vert", shaders_path + "/canvas.frag");
    float quadVertices[] = {
        boundings.x, boundings.w, 0.0f, 1.0f,
        boundings.x, boundings.y, 0.0f, 0.0f,
        boundings.z, boundings.w, 1.0f, 1.0f,
        boundings.z, boundings.y, 1.0f, 0.0f,
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);    
    glBindVertexArray(0);

    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture_size.x, texture_size.y, 0, GL_RGBA, GL_FLOAT, nullptr);
    glClearTexImage(texture, 0, GL_RGB, GL_FLOAT, &glm::vec4(0.5)[0]);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
}

void Canvas::update()
{

}

void Canvas::draw()
{
    glBindTexture(GL_TEXTURE_2D, texture);
    shader.use();
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}
