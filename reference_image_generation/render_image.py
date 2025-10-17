
from OpenGL.GL import *
from glfw.GLFW import *
    
from glfw import _GLFWwindow as GLFWwindow
    
import cv2

import glm

import numpy as np

from shader_m import Shader
from model import Model

import platform

def render_image(SCR_WIDTH, SCR_HEIGHT, buffer, fx, fy, cx, cy, R, tvec) -> int:
    glfwInit()
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)

    if (platform.system() == "Darwin"): # APPLE
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Synthetic Image", None, None)
    if (window == None):

        print("Failed to create GLFW window")
        glfwTerminate()
        return -1

    glfwMakeContextCurrent(window)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    glEnable(GL_DEPTH_TEST)

    ourShader = Shader("vertex_shader.vs", "fragment_shader.fs")
    # ourModel = Model("/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/athena-demo-resources/obj files/bunny_no_markers.obj")
    ourModel = Model("/Users/artsloan/Dropbox/Documents/Code/research/ARES/ATHENA/athena-demo-resources/obj files/3dbenchy.obj")

    near = 0.1
    far = 1000.0

    K = glm.mat4(1.0)
    K[0, 0] = fx
    K[1, 1] = fy
    K[2, 0] = cx
    K[2, 1] = cy

    cv2gl = glm.mat4(1.0)
    cv2gl[1, 1] = -1.0
    cv2gl[2, 2] = -1.0

    model = glm.mat4(1.0)
    model[0, 0] = R[0,0]   
    model[1, 0] = R[0,1] 
    model[2, 0] = R[0,2] 
    model[0, 1] = R[1,0] 
    model[1, 1] = R[1,1]  
    model[2, 1] = R[1,2] 
    model[0, 2] = R[2,0]  
    model[1, 2] = R[2,1]   
    model[2, 2] = R[2,2] 
    
    view = glm.mat4(1.0)
    view[3, 0] = tvec[0,0] 
    view[3, 1] = tvec[1,0] 
    view[3, 2] = tvec[2,0] 

    projection = glm.mat4()
    projection[0, 0] = 2 * K[0][0] / SCR_WIDTH 
    projection[1, 0] = 0.0                    
    projection[2, 0] = (SCR_WIDTH - 2.0 * K[2][0]) / SCR_WIDTH
    projection[3, 0] = 0.0
    projection[0, 1] = 0.0                    
    projection[1, 1] = -2.0 * K[1][1] / SCR_HEIGHT   
    projection[2, 1] = (SCR_HEIGHT - 2.0 * K[2][1]) / SCR_HEIGHT 
    projection[3, 1] = 0.0
    projection[0, 2] = 0.0                    
    projection[1, 2] = 0.0                       
    projection[2, 2] = (-1.0 * far - near) / (far - near)    
    projection[3, 2] = -2.0 * far * near / (far - near)
    projection[0, 3] = 0.0                    
    projection[1, 3] = 0.0                       
    projection[2, 3] = -1.0                           
    projection[3, 3] = 0.0

    light_position = glm.vec3(0.25, 0.25, -0.25)

    while (not glfwWindowShouldClose(window)):

        processInput(window)

        glClearColor(0.05, 0.05, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        ourShader.use()

        ourShader.setMat4("model", model)
        ourShader.setMat4("view", view)
        ourShader.setMat4("cv2gl", cv2gl)
        ourShader.setMat4("projection", projection)
        ourShader.setVec3("lightPos", light_position)
        ourShader.setVec3("lightColor", 1.0, 1.0, 1.0)
        ourShader.setVec3("objectColor", 1.0, 0.5, 0.31)

        ourModel.Draw(ourShader)
        glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, buffer)

        glfwSwapBuffers(window)
        glfwSetWindowShouldClose(window, GLFW_TRUE)
        glfwPollEvents()

    glfwTerminate()
    return 0

def processInput(window: GLFWwindow) -> None:

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS):
        glfwSetWindowShouldClose(window, True)

def framebuffer_size_callback(window: GLFWwindow, width: int, height: int) -> None:

    glViewport(0, 0, width, height)


if __name__ == '__main__':
    #Example usage
    width, height, channels = 960, 540, 3
    buffer = bytearray(width * height * channels)
    R = np.array([[ 0.99908534, -0.01708465,  0.03919934],
                  [ 0.0139356,  -0.73657653, -0.67621063],
                  [ 0.04042614,  0.67613839 ,-0.73566473]])
    tvec = np.array([[-0.0113534 ],
                     [-0.01899627],
                     [ 0.41404239]])
    
    render_image(width, height, buffer,2376.82,2364.56,524.973,356.555, R, tvec)

    image_data = np.frombuffer(buffer, dtype=np.uint8)
    image = image_data.reshape((540, 960, 3))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('OpenGL Capture', image)
    cv2.imwrite('synthetic_image.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

