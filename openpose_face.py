import numpy as np
import ctypes as ct
import cv2
import os

import FaceRendering
import utils
import models
import NonLinearLeastSquares
import ImageProcessing

from sys import platform
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    os.environ['PATH'] = dir_path + "/bin;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/lib;" + os.environ['PATH']
class OpenPose(object):
    """
    Ctypes linkage
    """
    if platform == "linux" or platform == "linux2":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.so')
    elif platform == "darwin":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.dylib')
    elif platform == "win32":
        try:
            _libop= np.ctypeslib.load_library('_openpose', 'D:/Tensor/FaceSwap-master/lib/_openpose.dll')
        except OSError as e:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Debug/_openpose.dll')
    _libop.newOP.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float, ct.c_float, ct.c_int, ct.c_float, ct.c_float, ct.c_int, ct.c_bool, ct.c_char_p]
    _libop.newOP.restype = ct.c_void_p
    _libop.delOP.argtypes = [ct.c_void_p]
    _libop.delOP.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.uint8), ct.c_bool]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    _libop.getfaceOutputs.argtypes = [ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getfaceOutputs.restype = None
    

    _libop.poseFromHeatmap.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.poseFromHeatmap.restype = None

    def encode(self, string):
        return ct.c_char_p(string.encode('utf-8'))

    def __init__(self, params):
        """
        OpenPose Constructor: Prepares OpenPose object

        Parameters
        ----------
        params : dict of required parameters. refer to openpose example for more details

        Returns
        -------
        outs: OpenPose object
        """
        self.op = self._libop.newOP(params["logging_level"],
		                            self.encode(params["output_resolution"]),
                                    self.encode(params["net_resolution"]),
                                    self.encode(params["model_pose"]),
                                    params["alpha_pose"],
                                    params["scale_gap"],
                                    params["scale_number"],
                                    params["render_threshold"],
                                    params["face_render_threshold"],
                                    params["num_gpu_start"],
                                    params["disable_blending"],
                                    self.encode(params["default_model_folder"]))

    def __del__(self):
        """
        OpenPose Destructor: Destroys OpenPose object
        """
        self._libop.delOP(self.op)

    def forward(self, image, display = False):
        """
        Forward: Takes in an image and returns the human 2D poses, along with drawn image if required

        Parameters
        ----------
        image : color image of type ndarray
        display : If set to true, we return both the pose and an annotated image for visualization

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(3),dtype=np.int32)
        size2 = np.zeros(shape=(3),dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], size, size2, displayImage, display)

        array = np.zeros(shape=(size),dtype=np.float32)        
        array2 = np.zeros(shape=(size2),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        self._libop.getfaceOutputs(self.op, array2)

        return array, array2 ,displayImage

    def poseFromHM(self, image, hm, ratios=[1]):
        """
        Pose From Heatmap: Takes in an image, computed heatmaps, and require scales and computes pose

        Parameters
        ----------
        image : color image of type ndarray
        hm : heatmap of type ndarray with heatmaps and part affinity fields
        ratios : scaling ration if needed to fuse multiple scales

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        if len(ratios) != len(hm):
            raise Exception("Ratio shape mismatch")

        # Find largest
        hm_combine = np.zeros(shape=(len(hm), hm[0].shape[1], hm[0].shape[2], hm[0].shape[3]),dtype=np.float32)
        i=0
        for h in hm:
           hm_combine[i,:,0:h.shape[2],0:h.shape[3]] = h
           i+=1
        hm = hm_combine

        ratios = np.array(ratios,dtype=np.float32)

        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(4),dtype=np.int32)
        size[0] = hm.shape[0]
        size[1] = hm.shape[1]
        size[2] = hm.shape[2]
        size[3] = hm.shape[3]

        self._libop.poseFromHeatmap(self.op, image, shape[0], shape[1], displayImage, hm, size, ratios)
        array = np.zeros(shape=(size[0],size[1],size[2]),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        return array, displayImage

    @staticmethod
    def process_frames(frame, boxsize = 368, scales = [1]):
        base_net_res = None
        imagesForNet = []
        imagesOrig = []
        for idx, scale in enumerate(scales):
            # Calculate net resolution (width, height)
            if idx == 0:
                net_res = (16 * int((boxsize * frame.shape[1] / float(frame.shape[0]) / 16) + 0.5), boxsize)
                base_net_res = net_res
            else:
                net_res = ((min(base_net_res[0], max(1, int((base_net_res[0] * scale)+0.5)/16*16))),
                          (min(base_net_res[1], max(1, int((base_net_res[1] * scale)+0.5)/16*16))))
            input_res = [frame.shape[1], frame.shape[0]]
            scale_factor = min((net_res[0] - 1) / float(input_res[0] - 1), (net_res[1] - 1) / float(input_res[1] - 1))
            warp_matrix = np.array([[scale_factor,0,0],
                                    [0,scale_factor,0]])
            if scale_factor != 1:
                imageForNet = cv2.warpAffine(frame, warp_matrix, net_res, flags=(cv2.INTER_AREA if scale_factor < 1. else cv2.INTER_CUBIC), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            else:
                imageForNet = frame.copy()

            imageOrig = imageForNet.copy()
            imageForNet = imageForNet.astype(float)
            imageForNet = imageForNet/256. - 0.5
            imageForNet = np.transpose(imageForNet, (2,0,1))

            imagesForNet.append(imageForNet)
            imagesOrig.append(imageOrig)

        return imagesForNet, imagesOrig

    @staticmethod
    def draw_all(imageForNet, heatmaps, currIndex, div=4., norm=False):
        netDecreaseFactor = float(imageForNet.shape[0]) / float(heatmaps.shape[2]) # 8
        resized_heatmaps = np.zeros(shape=(heatmaps.shape[0], heatmaps.shape[1], imageForNet.shape[0], imageForNet.shape[1]))
        num_maps = heatmaps.shape[1]
        combined = None
        for i in range(0, num_maps):
            heatmap = heatmaps[0,i,:,:]
            resizedHeatmap = cv2.resize(heatmap, (0,0), fx=netDecreaseFactor, fy=netDecreaseFactor)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resizedHeatmap)

            if i==currIndex and currIndex >=0:
                resizedHeatmap = np.abs(resizedHeatmap)
                resizedHeatmap = (resizedHeatmap*255.).astype(dtype='uint8')
                im_color = cv2.applyColorMap(resizedHeatmap, cv2.COLORMAP_JET)
                resizedHeatmap = cv2.addWeighted(imageForNet, 1, im_color, 0.3, 0)
                cv2.circle(resizedHeatmap, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
                return resizedHeatmap
            else:
                resizedHeatmap = np.abs(resizedHeatmap)
                if combined is None:
                    combined = np.copy(resizedHeatmap);
                else:
                    if i <= num_maps-2:
                        combined += resizedHeatmap;
                        if norm:
                            combined = np.maximum(0, np.minimum(1, combined));

        if currIndex < 0:
            combined /= div
            combined = (combined*255.).astype(dtype='uint8')
            im_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(imageForNet, 0.5, im_color, 0.5, 0)
            cv2.circle(combined, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
            return combined


if __name__ == "__main__":
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["face_render_threshold"] = 0.4
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] = "models/"
    openpose = OpenPose(params)


    image_name = "data/hangain.jpg"
    maxImageSizeForDetection = 320

    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(predictor_path)
    mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("candide.npz")
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

    modelParams = None
    lockedTranslation = False
    drawOverlay = False
    cap = cv2.VideoCapture(0)
    writer = None
    cameraImg = cap.read()[1]

    textureImg = cv2.imread(image_name)
    textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, openpose)
    renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)
    #arr, arr2, output_image = openpose.forward(img, False)
    while True:
        cameraImg = cap.read()[1]
        cameraImg = cv2.flip(cameraImg, 1)
        arr, shapes2D, output_image = openpose.forward(cameraImg, False)
        shapes2D = shapes2D[:,:,:2]
        if shapes2D is not None:
            for shape2D in shapes2D:
                #3D model parameter initialization                
                count = 0;
                shape2D = shape2D.T

                for x in  shape2D[0]:                    
                    if x <= 0.0 or x >= cameraImg.shape[1] :
                        count = count +1
                if(count< 35) :
                    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

                    #3D model parameter optimization
                    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

                    #rendering the model to an image
                    shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
                    renderedImg = renderer.render(shape3D)

                    #blending of the rendered face with the image
                    mask = np.copy(renderedImg[:, :, 0])
                    renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
                    cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)
       

                    #drawing of the mesh and keypoints
                    if drawOverlay:
                        drawPoints(cameraImg, shape2D.T)
                        drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

        if writer is not None:
            writer.write(cameraImg)

        cv2.imshow('image', cameraImg)
        key = cv2.waitKey(1)

        if key == 27:
            break
        if key == ord('q'):
            break
        if key == ord('t'):
            drawOverlay = not drawOverlay
        if key == ord('r'):
            if writer is None:
                print("Starting video writer")
                writer = cv2.VideoWriter("../out.avi", cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'), 25, (cameraImg.shape[1], cameraImg.shape[0]))

                if writer.isOpened():
                    print( "Writer succesfully opened")
                else:
                    writer = None
                    print ("Writer opening failed")
            else:
                print ("Stopping video writer")
                writer.release()
                writer = None


