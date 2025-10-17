from OpenGL.GL import * 
    
import glm

from shader import Shader

import ctypes

from typing import List

class Vertex:
    PositionOffset = ctypes.c_void_p(0)
    NormalOffset = ctypes.c_void_p(glm.sizeof(glm.vec3))
    size = NormalOffset.value + glm.sizeof(glm.vec3)

class Mesh:

    def __init__(self, vertices : glm.array, indices : List[int]):
        self.vertices = vertices
        self.indices = glm.array.from_numbers(glm.uint32, *indices)

        self.setupMesh()

    def Draw(self, shader : Shader): 

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def setupMesh(self):

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices.ptr, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices.ptr, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)	
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, Vertex.size, None)
        glEnableVertexAttribArray(1)	
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, Vertex.size, Vertex.NormalOffset)

        glBindVertexArray(0)

