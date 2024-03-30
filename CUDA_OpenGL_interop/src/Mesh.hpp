#pragma once
#include "Shader.hpp"
#include <glm/glm.hpp>
#include <vector>
#include "DataTypes.hpp"



class Mesh {
public:
    // mesh data
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture>      textures;

    Mesh() {}
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures);
    void setupMesh();
    void updateMesh();
    void draw(Shader& shader, GLenum mode = GL_TRIANGLES);
protected:
    //  render data
    GLuint VAO, VBO, EBO;
};