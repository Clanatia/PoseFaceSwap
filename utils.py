import numpy as np
import cv2
import models
import NonLinearLeastSquares

def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ

def flipWinding(triangle):
    return [triangle[1], triangle[0], triangle[2]]

def fixMeshWinding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = getNormal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flipWinding(triangle)

    return mesh

def getShape3D(mean3DShape, blendshapes, params):
    #skalowanie
    s = params[0]
    #rotacja
    r = params[1:4]
    #przesuniecie (translacja)
    t = params[4:6]
    w = params[6:]

    #macierz rotacji z wektora rotacji, wzor Rodriguesa
    R = cv2.Rodrigues(r)[0]
    shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D

def getMask(renderedImg):
    mask = np.zeros(renderedImg.shape[:2], dtype=np.uint8)

def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    mean3DShape = faceModelFile["mean3DShape"]
    mesh = faceModelFile["mesh"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    blendshapes = faceModelFile["blendshapes"]
    mesh = fixMeshWinding(mesh, mean3DShape)

    return mean3DShape, blendshapes, mesh, idxs3D, idxs2D

class Point(object):
  def __init(self,x,y):
    self._x = x
    self._y = y
  @property
  def x(self):
    return self._x
  @property
  def y(self):
    return self._y

class Rectangle(object):
  def __init(self,x,y,width,height):
    self._x = x
    self._y = y
    self._width = width
    self._height = height
  @property
  def x(self):
    return self._x
  @property
  def y(self):
    return self._y
  @property
  def width(self):
    return self._width
  @property
  def height(self):
    return self._height


def getFaceFromPoseKeypoints(poseKeypoints, personIndex, neck, headNose, lEar, rEar, lEye, rEye, threshold):

    pointTopLeft = Point()
    faceSize = 0.0

    posePtr = poseKeypoints.at(personIndex*poseKeypoints.getSize(1)*poseKeypoints.getSize(2))
    neckScoreAbove = (posePtr[neck*3+2] > threshold)
    headNoseScoreAbove = (posePtr[headNose*3+2] > threshold)
    lEarScoreAbove = (posePtr[lEar*3+2] > threshold)
    rEarScoreAbove = (posePtr[rEar*3+2] > threshold)
    lEyeScoreAbove = (posePtr[lEye*3+2] > threshold)
    rEyeScoreAbove = (posePtr[rEye*3+2] > threshold)

    counter = 0
    # Face and neck given (e.g. MPI)
    if (headNose == lEar and lEar == rEar ) :
        if (neckScoreAbove and headNoseScoreAbove):
            pointTopLeft.x = posePtr[headNose*3];
            pointTopLeft.y = posePtr[headNose*3+1];
            faceSize = 1.33 * getDistance(poseKeypoints, personIndex, neck, headNose);

    # Face as average between different body keypoints (e.g. COCO)
    else :
        # factor * dist(neck, headNose)
        if (neckScoreAbove and headNoseScoreAbove) :
            #If profile (i.e. only 1 eye and ear visible) --> avg(headNose, eye & ear position)
            if ((lEyeScoreAbove) == (lEarScoreAbove) and (rEyeScoreAbove) == (rEarScoreAbove) and (lEyeScoreAbove) != (rEyeScoreAbove)) :
                if (lEyeScoreAbove) :
                    pointTopLeft.x += (posePtr[lEye*3] + posePtr[lEar*3] + posePtr[headNose*3]) / 3.0
                    pointTopLeft.y += (posePtr[lEye*3+1] + posePtr[lEar*3+1] + posePtr[headNose*3+1]) / 3.0
                    faceSize += 0.85 * (getDistance(poseKeypoints, personIndex, headNose, lEye) + getDistance(poseKeypoints, personIndex, headNose, lEar) + getDistance(poseKeypoints, personIndex, neck, headNose))
                else :#// if(lEyeScoreAbove)
                    pointTopLeft.x += (posePtr[rEye*3] + posePtr[rEar*3] + posePtr[headNose*3]) / 3.0
                    pointTopLeft.y += (posePtr[rEye*3+1] + posePtr[rEar*3+1] + posePtr[headNose*3+1]) / 3.0
                    faceSize += 0.85 * (getDistance(poseKeypoints, personIndex, headNose, rEye) + getDistance(poseKeypoints, personIndex, headNose, rEar) + getDistance(poseKeypoints, personIndex, neck, headNose))

            #else --> 2 * dist(neck, headNose)
            else :
                pointTopLeft.x += (posePtr[neck*3] + posePtr[headNose*3]) / 20
                pointTopLeft.y += (posePtr[neck*3+1] + posePtr[headNose*3+1]) / 2.0
                faceSize += 2.0 * getDistance(poseKeypoints, personIndex, neck, headNose)

            counter+=1

        #3 * dist(lEye, rEye)
        if (lEyeScoreAbove and rEyeScoreAbove) :
            pointTopLeft.x += (posePtr[lEye*3] + posePtr[rEye*3]) / 2.0
            pointTopLeft.y += (posePtr[lEye*3+1] + posePtr[rEye*3+1]) / 2.0
            faceSize += 3.0 * getDistance(poseKeypoints, personIndex, lEye, rEye)
            counter+=1
        # 2 * dist(lEar, rEar)
        if lEarScoreAbove and rEarScoreAbove :        
            pointTopLeft.x += (posePtr[lEar*3] + posePtr[rEar*3]) / 2.0
            pointTopLeft.y += (posePtr[lEar*3+1] + posePtr[rEar*3+1]) / 2.0
            faceSize += 2.0 * getDistance(poseKeypoints, personIndex, lEar, rEar)
            counter+=1
        # Average (if counter > 0)
        if counter > 0 :
            pointTopLeft /= float(counter)
            faceSize /= float(counter)
    
    return Rectangle(pointTopLeft.x - faceSize / 2, pointTopLeft.y - faceSize / 2, faceSize, faceSize)
       

def getFaceTextureCoords(img, mean3DShape, blendshapes, idxs2D, idxs3D, openpose):
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])
    arr, arr2, output_image = openpose.forward(img, False)
    for shape2D in arr2:
        keypoints = shape2D[:,:2].T
  
    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]), verbose=0)
    textureCoords = projectionModel.fun([mean3DShape, blendshapes], modelParams)

    return textureCoords