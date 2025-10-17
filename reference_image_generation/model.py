from OpenGL.GL import * 
import glm
import numpy as np
import assimp_py
from mesh import Mesh
from shader import Shader
import os


class Model:

    def __init__(self, path: str, gamma : bool = False):
        self.gammaCorrection = gamma
        self.loadModel(path)

    def Draw(self, shader : Shader):
        for mesh in self.meshes:
            mesh.Draw(shader)
            
    def loadModel(self, path : str):

        assimp_flags = assimp_py.Process_Triangulate | assimp_py.Process_GenSmoothNormals | assimp_py.Process_FlipUVs | assimp_py.Process_CalcTangentSpace
        scene = assimp_py.import_file(path, assimp_flags)
        self.directory = os.path.dirname(path)
        self.meshes = []
        self.processMeshes(scene)

    def processMeshes(self, scene : assimp_py.Scene):

        for mesh in scene.meshes:
            verts = mesh.vertices
            self.meshes.append(self.processMesh(mesh, scene))

    def processMesh(self, mesh : assimp_py.Mesh, scene : assimp_py.Scene) -> Mesh:
        indices = []
        # reshape the vertices, which is a flat array to a list of 3D points
        vertices = np.array(mesh.vertices).reshape(-1, 3)
        vertices *= 0.001 # Rescale
        # if there are normals, reshape them too
        if mesh.normals:
            normals = np.asarray(mesh.normals).reshape(-1, 3)
        else:
            normals = np.zeros_like(vertices)
        # interleave vertices and normals
        vertices = np.hstack((vertices, normals)).flatten().tolist()

        indices = mesh.indices.tolist()

        return Mesh(glm.array.from_numbers(glm.float32, *vertices), indices)
  
