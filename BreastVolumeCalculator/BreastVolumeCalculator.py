import logging
import os
import slicer
import vtk
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np


#
# BreastVolumeCalculator
#

class BreastVolumeCalculator(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "BreastVolumeCalculator"
        self.parent.categories = ["Quantification"]
        self.parent.dependencies = []
        self.parent.contributors = ["Pian Kexin (Wuhan University)"]
        self.parent.helpText = """This module calculates the size of a breast implant from breast MRI data"""
        self.parent.acknowledgementText = """This module was written by Pian Kexin from Wuhan University. """
        slicer.app.connect("startupCompleted()", self.registerSampleData)

    def registerSampleData(self):
        # Add data set to Sample Data module
        iconsPath = os.path.join(os.path.dirname(self.parent.path), 'Resources/Icons')
        import SampleData
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            category='Breast Implant Analyzer',
            sampleName='MRBreastImplant',
            uris='https://github.com/lancelevine/SlicerBreastImplantAnalyzer/raw/master/SampleData/MRBreastImplant.nrrd',
            fileNames='MRBreastImplant.nrrd',
            nodeNames='MRBreastImplant',
            thumbnailFileName=os.path.join(iconsPath, 'MRBreastImplant.png'),
            checksums='SHA256:4bbad3e4034005ddb06ac819bfae2ded2175838f733dfd6ee12f81787450258a'
        )


#
# BreastVolumeCalculatorWidget
#

class BreastVolumeCalculatorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.inputVolume = None
        self.logic = None
        self._parameterNode = None
        self.placeModePersistence = 1
        self.implantSegmentId = None
        self.segmentEditorNode = None
        self.segmentationNodeObserverTag = None
        self.segmentationNode = None
        self.segmentationMask = None

    def setup(self):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer)
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/BreastVolumeCalculator.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create a new parameterNode
        # This parameterNode stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.logic = BreastVolumeCalculatorLogic()

        self.selectParameterNode()

        # Connections
        self.ui.getResultButton.connect('clicked(bool)', self.getResultButton)
        self.ui.contrastButton.connect('toggled(bool)', self.onContrastButton)
        self.ui.markSegButton.connect('toggled(bool)', self.onMarkSegButton)
        self.ui.editSegButton.connect('clicked(bool)', self.onEditSegButton)
        self.ui.markupButton.connect('toggled(bool)', self.markupTest)
        self.ui.measureAngleButton.connect('clicked(bool)', self.ShowAngle)
        self.ui.curveButton.connect('clicked(bool)', self.addCurveTest)
        self.ui.lengthButton.connect('clicked(bool)', self.lengthTest)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Connect observers to scene events
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

        self.ui.resultLabel.text = ""
        self.ui.tipLabel.text = ""

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def cleanup(self):
        """
    Called when the application closes and the module widget is destroyed.
    """
        self.removeObservers()
        logging.info("cleaning up")

    def enter(self):
        self.selectParameterNode()

    def exit(self):
        self.stopFiducialPlacement()
        self.removeObservers()

    def selectParameterNode(self):
        # Ensure parameter node exists
        self.setParameterNode(self.logic.getParameterNode())

        # Select first volume node in the scene by default (if none is selected yet)
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

        self.updateGUIFromParameterNode()

    def onSceneStartClose(self, caller, event):
        self.stopFiducialPlacement()
        self._parameterNode = None
        self.ui.resultLabel.text = ""
        self.ui.tipLabel.text = ""

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.selectParameterNode()

    def onSceneEndImport(self, caller, event):
        if self.parent.isEntered:
            self.selectParameterNode()

    def setParameterNode(self, inputParameterNode):
        """
    Adds observers to the selected parameter node. Observation is needed because when the
    parameter node is changed then the GUI must be updated immediately.
    """
        if inputParameterNode == self._parameterNode:  # No change
            return

        # Unobserve previusly selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        if inputParameterNode is not None:
            self.addObserver(inputParameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

        # Disable all sections if no parameter node is selected
        self.ui.basicCollapsibleButton.enabled = self._parameterNode is not None
        if self._parameterNode is None:
            return

        # Update each widget from parameter node
        # Need to temporarily block signals to prevent infinite recursion (MRML node update triggers
        # GUI update, which triggers MRML node update, which triggers GUI update, ...)

        inputVolume = self._parameterNode.GetNodeReference("InputVolume")

        wasBlocked = self.ui.inputSelector.blockSignals(True)
        self.ui.inputSelector.setCurrentNode(inputVolume)
        self.ui.inputSelector.blockSignals(wasBlocked)

        # Update buttons states and tooltips
        self.ui.editSegButton.enabled = False
        self.ui.measureAngleButton.enabled = False
        self.ui.lengthButton.enabled = False
        self.ui.markupButton.enabled = False
        self.ui.curveButton.enabled = False
        self.ui.getResultButton.enabled = False

        if inputVolume:
            self.inputVolume = inputVolume
            self.ui.contrastButton.enabled = True
            self.ui.markSegButton.enabled = True
        else:
            self.ui.contrastButton.enabled = False
            self.ui.markSegButton.enabled = False

        wasBlocked = self.ui.contrastButton.blockSignals(True)
        self.ui.contrastButton.setChecked(self.logic.isOriginalContrastAvailable(inputVolume))
        self.ui.contrastButton.blockSignals(wasBlocked)

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """
        if self._parameterNode is None:
            return
        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)

    def startSegmentation(self):
        mask = self.logic.readPic(self.inputVolume)
        self.segmentationMask = mask
        inputVolume = self._parameterNode.GetNodeReference("InputVolume")
        self.ui.inputSelector.setCurrentNode(inputVolume)
        self.drawMask()

    def drawMask(self):
        print('====== create segmentation ======')
        # Create segmentation
        masterVolumeNode = self.ui.inputSelector.currentNode()
        segmentationNode = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(segmentationNode)
        segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)

        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)

        mask = self.segmentationMask
        X, Y, Z = mask.shape
        print('X,Y,Z')
        print(X, Y, Z)

        def coordinate_to_ras(c):    # 像素坐标coordinate到RAS的转化
            origin = np.array([184.8624, 117.2048, -32.4415])
            matrix = np.array([
                [-1.0000, 0.0000, 0.0000],
                [-0.0000, -0.9999, 0.0122],
                [0.0000, 0.0122, 0.9999]
            ])
            # space = [1.023, 1.023, 1.199]
            space = masterVolumeNode.GetSpacing()
            ras_c = c * space
            ras_c = np.matmul(matrix, ras_c)
            ras_c = ras_c + origin
            return ras_c

        def addSegmentation(segmentID, selectedSegmentID):
            segmentEditorNode.SetSelectedSegmentID(selectedSegmentID)
            segmentEditorWidget.setActiveEffectByName('Logical operators')
            effect = segmentEditorWidget.activeEffect()
            effect.setParameter("Operation", "UNION")
            effect.setParameter("BypassMasking", "1")
            effect.setParameter("ModifierSegmentID", segmentID)
            effect.self().onApply()

        firstSeed = vtk.vtkSphereSource()
        firstSeed.SetCenter(70, 20, 30)
        firstSeed.SetRadius(0)
        firstSeed.Update()
        firstSegmentID = segmentationNode.AddSegmentFromClosedSurfaceRepresentation(firstSeed.GetOutput(), "Breast",
                                                                                    [0.5, 0.8, 0.2])
        append = vtk.vtkAppendPolyData()

        for x in range(40, X-40, 5):   # 0, X, 1
            for y in range(0, Y, 5):   # 0, Y, 1
                for z in range(0, Z, 30):
                    if mask[x, y, z] == 1:

                        print('x, y, z')
                        print([x, y, z])
                        ras = np.array([y+179, x + 83, z+3])
                        coordinate = coordinate_to_ras(ras)
                        print('coordinate')
                        print(coordinate)
                        secondSeed = vtk.vtkSphereSource()
                        secondSeed.SetCenter(coordinate[0], coordinate[1], coordinate[2])
                        secondSeed.SetRadius(15)
                        secondSeed.Update()
                        append.AddInputData(secondSeed.GetOutput())
                        append.Update()
                        secondSegmentID = segmentationNode.AddSegmentFromClosedSurfaceRepresentation(append.GetOutput(),
                                                                                                     "second", [0.6, 0.1, 0.5])
                        addSegmentation(secondSegmentID, firstSegmentID)
                        segmentationNode.RemoveSegment(secondSegmentID)

                    else:
                        continue
                continue
            continue

        segmentEditorWidget.setActiveEffectByName('Grow from seeds')
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("Seed locality", "UNION")
        effect.self().onApply()


        slicer.mrmlScene.RemoveNode(segmentEditorNode)
        self.implantSegmentId = firstSegmentID
        self.segmentationNode = segmentationNode

    def stopFiducialPlacement(self):
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToViewTransformMode()
        interactionNode.SetPlaceModePersistence(0)

    def onContrastButton(self, pushed):
        self.logic.setAutoContrast(pushed, self.ui.inputSelector.currentNode())

    def getResultButton(self, calc):
        results = self.logic.computeImplantVolumeCc(self.ui.inputSelector.currentNode(), self.segmentationNode,
                                                    self.implantSegmentId)
        self.ui.resultLabel.text = "Volume: " + '{:.2f}'.format(results[0]) + "cm3\nSurface area: " \
                                   + '{:.2f}'.format(results[1]) + "mm2"

    def ShowAngle(self):
        lineNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode')
        lineNum = len(lineNodes)
        lineDirectionVectors = []
        lineNodeNames = []
        for lineNode in lineNodes:
            measurementName = lineNode.GetName()
            lineNodeNames.append(measurementName)
        for i in range(lineNum):
            lineNode = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsLineNode", lineNodeNames[i])
            lineStartPos = np.zeros(3)
            lineEndPos = np.zeros(3)
            lineNode.GetNthControlPointPositionWorld(0, lineStartPos)
            lineNode.GetNthControlPointPositionWorld(1, lineEndPos)
            lineDirectionVector = (lineEndPos - lineStartPos) / np.linalg.norm(lineEndPos - lineStartPos)
            lineDirectionVectors.append(lineDirectionVector)
        text = ''
        for i in range(lineNum):
            for j in range(i + 1, lineNum):
                angleRad = vtk.vtkMath.AngleBetweenVectors(lineDirectionVectors[i], lineDirectionVectors[j])
                angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
                textTemp = 'Angle between lines ' + str(lineNodeNames[i]) + ' and ' + str(
                    lineNodeNames[j]) + ' is: ' + str(round(angleDeg, 3)) + '\n'
                text += textTemp
        self.ui.tipLabel.text = text

    def addCurveTest(self):
        pointListNode = slicer.util.getNode("F")
        numFids = pointListNode.GetNumberOfFiducials()
        curve2Pos = []
        for i in range(numFids):
            ras = [0, 0, 0]
            pointListNode.GetNthFiducialPosition(i, ras)
            # the world position is the RAS position with any transform matrices applied
            world = [0, 0, 0, 0]
            pointListNode.GetNthFiducialWorldCoordinates(i, world)
            curve2Pos.append(ras)
        curve2PosArray = np.array(curve2Pos)
        # Create curve from numpy array
        curveNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        slicer.util.updateMarkupsControlPointsFromArray(curveNode2, curve2PosArray)

    def lengthTest(self):
        curveLengths = []
        curveNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode")
        text = ''
        for curve in curveNodes:
            measurementName = curve.GetName()
            curve.GetMeasurement('length').SetEnabled(True)
            length = str(curve.GetMeasurement('length').GetValue())
            curveLengths.append('  '.join([measurementName, length]))
            text += '  '.join([measurementName, length])
            text += '\n'
        self.ui.tipLabel.text = text

    def onMarkSegButton(self, mark):
        if mark:
            self.ui.editSegButton.enabled = True
            self.ui.measureAngleButton.enabled = True
            self.ui.lengthButton.enabled = True
            self.ui.curveButton.enabled = True
            self.ui.markupButton.enabled = True
            self.ui.getResultButton.enabled = True

            self.ui.markSegButton.text = "Cancel"
            self.ui.tipLabel.text = "Draw segmentation automatically. You can edit the segmentation. " \
                                    "\nClick the button again to cancel the drawing and clear the segmentation."

            self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
            segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            self.segmentEditorWidget = segmentEditorWidget
            masterVolumeNode = self.ui.inputSelector.currentNode()
            self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            segmentEditorNode = self.segmentEditorNode
            slicer.mrmlScene.AddNode(segmentEditorNode)
            self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            # self.segmentEditorWidget.setSegmentationNode(self.segmentationNode)
            self.segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)

            # Start drawing segmentations
            self.startSegmentation()
        else:
            self.stopMarkSeg()
            self.ui.tipLabel.text = ""

    def stopMarkSeg(self):
        self.ui.editSegButton.enabled = False
        self.ui.measureAngleButton.enabled = False
        self.ui.lengthButton.enabled = False
        self.ui.curveButton.enabled = False
        self.ui.markupButton.enabled = False
        self.ui.getResultButton.enabled = False

        # Stop requested
        self.ui.markSegButton.text = "Mark the Segmentation"
        self.ui.resultLabel.text = ""

        # Clean up
        if self.segmentEditorNode:
            slicer.mrmlScene.RemoveNode(self.segmentEditorNode)
        if self.implantSegmentId:
            self.segmentationNode.RemoveSegment(self.implantSegmentId)

    def onEditSegButton(self):
        self.segmentEditorWidget.setSegmentationNode(self.segmentationNode)
        # Create segment editor to get access to effects
        self.segmentEditorWidget.show()

    def markupTest(self, add):
        if add:
            self.ui.markupButton.text = "Cancel"
            placeModePersistence = self.placeModePersistence
            slicer.modules.markups.logic().StartPlaceMode(placeModePersistence)
            self.ui.tipLabel.text = "Click the mouse to add control points. Click again to stop adding."
        else:
            selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
            selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
            self.ui.markupButton.text = "Add markups test"
            interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
            interactionNode.SwitchToViewTransformMode()
            interactionNode.SetPlaceModePersistence(0)
            self.ui.tipLabel.text = ""
        print('end markupTest')


#
# BreastVolumeCalculatorLogic
#

class BreastVolumeCalculatorLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.segmentationNode = None

    def isOriginalContrastAvailable(self, volumeNode):
        if not volumeNode:
            return False
        originalWindow = volumeNode.GetAttribute("BreastVolumeCalculator.OriginalWindow")
        originalLevel = volumeNode.GetAttribute("BreastVolumeCalculator.OriginalLevel")
        return originalWindow and originalLevel

    def setAutoContrast(self, enable, volumeNode):
        displayNode = volumeNode.GetDisplayNode()
        if enable:
            # Save original window/level to volume node
            volumeNode.SetAttribute("BreastVolumeCalculator.OriginalWindow", str(displayNode.GetWindow()))
            volumeNode.SetAttribute("BreastVolumeCalculator.OriginalLevel", str(displayNode.GetLevel()))
            # force recomputation of window/level
            displayNode.AutoWindowLevelOff()
            displayNode.AutoWindowLevelOn()
        else:
            # restore original window/level
            if not self.isOriginalContrastAvailable(volumeNode):
                raise ValueError(
                    "Failed to restore original window/level for volume node, previous values were not found")
            originalWindow = volumeNode.GetAttribute("BreastVolumeCalculator.OriginalWindow")
            originalLevel = volumeNode.GetAttribute("BreastVolumeCalculator.OriginalLevel")
            # Remove original values so that the GUI knows that the original values are used now
            volumeNode.SetAttribute("BreastVolumeCalculator.OriginalWindow", "")
            volumeNode.SetAttribute("BreastVolumeCalculator.OriginalLevel", "")
            displayNode.AutoWindowLevelOff()
            displayNode.SetWindow(float(originalWindow))
            displayNode.SetLevel(float(originalLevel))

    def setDefaultParameters(self, parameterNode):
        """
    Initialize parameter node with default settings.
    """
        if not parameterNode.GetParameter("SeedLocality"):
            parameterNode.SetParameter("SeedLocality", "0.0")

    def computeImplantVolumeCc(self, inputVolume, segmentationNode, implantSegmentId):
        masterVolumeNode = inputVolume

        if segmentationNode:
            # Compute segment volumes
            import SegmentStatistics
            segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
            segStatLogic.getParameterNode().SetParameter("Segmentation", segmentationNode.GetID())
            segStatLogic.getParameterNode().SetParameter("ScalarVolume", masterVolumeNode.GetID())
            segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.enabled", "False")
            segStatLogic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.voxel_count.enabled",
                                                         "False")
            segStatLogic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.volume_mm3.enabled",
                                                         "False")
            segStatLogic.computeStatistics()
            print(segStatLogic.getStatistics())
            implantVolumeCc = segStatLogic.getStatistics()[
                implantSegmentId, 'ScalarVolumeSegmentStatisticsPlugin.volume_cm3']
            implantSurfaceCc = segStatLogic.getStatistics()[
                implantSegmentId, 'ClosedSurfaceSegmentStatisticsPlugin.surface_mm2']
            logging.info("Processing result: " + str(implantVolumeCc))
            results = [implantVolumeCc, implantSurfaceCc]
            return results

    def readPic(self, inputVolume):
        import pydicom
        from pydicom.filereader import dcmread
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            slicer.util.pip_install("matplotlib")
            import matplotlib
        import matplotlib.pyplot as plt
        try:
            import cv2
        except ModuleNotFoundError:
            slicer.util.pip_install("opencv-python")
            import cv2
        import torch
        import itertools as itt
        import sys
        import os

        erosion_matrix = np.array([  # 用于表示该点所代表的正方体有多少属于半径为2的球内
            [
                [0.00, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.05, 0.20, 0.05, 0.00],
                [0.00, 0.20, 0.46, 0.20, 0.00],
                [0.00, 0.05, 0.20, 0.05, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00]],
            [
                [0.00, 0.05, 0.20, 0.05, 0.00],
                [0.05, 0.20, 0.98, 0.20, 0.05],
                [0.20, 0.98, 1.00, 0.98, 0.20],
                [0.05, 0.20, 0.98, 0.20, 0.05],
                [0.00, 0.05, 0.20, 0.05, 0.00]],
            [
                [0.00, 0.20, 0.46, 0.20, 0.00],
                [0.20, 0.98, 1.00, 0.98, 0.20],
                [0.46, 1.00, 1.00, 1.00, 0.46],
                [0.20, 0.98, 1.00, 0.98, 0.20],
                [0.00, 0.20, 0.46, 0.20, 0.00]],
            [
                [0.00, 0.05, 0.20, 0.05, 0.00],
                [0.05, 0.20, 0.98, 0.20, 0.05],
                [0.20, 0.98, 1.00, 0.98, 0.20],
                [0.05, 0.20, 0.98, 0.20, 0.05],
                [0.00, 0.05, 0.20, 0.05, 0.00]],
            [
                [0.00, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.05, 0.20, 0.05, 0.00],
                [0.00, 0.20, 0.46, 0.20, 0.00],
                [0.00, 0.05, 0.20, 0.05, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00]]
        ])
        erosion_matrix /= erosion_matrix.sum()

        kernel_dx = np.zeros([3, 3, 3])  # 前胸->后背(切片中上->下)
        kernel_dx[0, 1, 1] = 0.5
        kernel_dx[-1, 1, 1] = -0.5
        kernel_dy = kernel_dx.swapaxes(0, 1)  # 左臂->右臂(切片中左->右)  swapaxes：转置矩阵
        kernel_dz = kernel_dx.swapaxes(0, 2)  # 头部->腹部(早切片->晚切片)
        instUids = inputVolume.GetAttribute("DICOM.instanceUIDs").split()

        def pathFromNode(node):
            storageNode = node.GetStorageNode()
            if storageNode is not None:  # loaded via drag-drop
                filepath = storageNode.GetFullNameFromFileName()
            else:  # Loaded via DICOM browser
                filepath = slicer.dicomDatabase.fileForInstance(instUids[0])
            return filepath

        path = pathFromNode(inputVolume)
        ref_ds = pydicom.read_file(path)
        parent_path = os.path.dirname(path)
        imgs = []
        for root, dirs, files in os.walk(parent_path):
            imgs = files[:-1]
        for i in range(len(imgs)):
            imgs[i] = parent_path + '\\' + imgs[i]

        Pixel_Dims = (int(ref_ds.Rows), int(ref_ds.Columns), len(imgs))  # 三维数组：行数，列数，分层数
        ArrayDicom = np.zeros(Pixel_Dims, dtype=ref_ds.pixel_array.dtype)

        for i, img in zip(range(len(imgs)), imgs):  # zip：两个数组配对，一一对应
            ds = pydicom.read_file(img)  # read_file = dcmread(path)
            ArrayDicom[:, :, i] = ds.pixel_array  # [:, :, i]取矩阵平面的列元素
        ArrayDicom = ArrayDicom.astype(np.float64)

        # 剪切到合适尺寸  352x352x112 -> 120x120x112(x2)
        Left = ArrayDicom[80:200, 56:176, :]
        Right = np.flip(ArrayDicom[80:200, 176:296, :], 1)  # 右胸翻转到左胸位置
        test = Right

        # 模糊：降低图像的噪声和减少图像的细节。
        test = self.conv3d(test, self.gauss(5, 1.5))  # 120x120x112 -> 116x116x108

        # 阈值切割
        edge, mask, sample_points = self.thresholdLowerCut(test[1:-1, 1:-1, 1:-1], 25)  # 裁边至114x114x106与后面的grad匹配

        # 梯度
        grad_x, grad_y, grad_z = [self.conv3d(test, d) for d in [kernel_dx, kernel_dy, kernel_dz]]
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)  # 114x114x106


        # 扩张边缘
        X, Y, Z = grad.shape
        wall_len = 6  # 标记墙厚度
        inf_range = 3  # 影响范围
        for x, y, z in sample_points:
            flag = False
            for a, r in zip([x, y, z], [X, Y, Z]):  # 检查xyz是否在合法范围中
                if a < inf_range or a >= r - inf_range:
                    flag = True
                    break
            if flag:
                continue
            dx, dy, dz, g = grad_x[x, y, z], grad_y[x, y, z], grad_z[x, y, z], grad[x, y, z]
            # product 以元组的形式，根据输入的可遍历对象生成笛卡尔积
            for i, j, k in itt.product(np.arange(-inf_range, inf_range + 1), repeat=3):
                dist = abs(i * dx + j * dy + k * dz) / g  # (x+i, y+j, z+k)点到(x, y, z)点在梯度方向投影的距离(此点到等势面的距离)
                mark_value = max(wall_len - dist, 0) / wall_len  # 标记墙在距离等势面r单位的位置处线性衰减为0
                edge[x + i, y + j, z + k] = max(edge[x + i, y + j, z + k], mark_value)


        # 非极大值抑制
        timer = 0
        for x, y, z in itt.product(range(1, X - 1), range(1, Y - 1), range(1, Z - 1)):
            timer += 1
            if timer >= 100000:
                timer = 0
            grad_p = grad[x, y, z]
            if mask[x, y, z] == 0:
                continue
            dx, dy, dz = grad_x[x, y, z] / grad_p, grad_y[x, y, z] / grad_p, grad_z[x, y, z] / grad_p
            neighbors = [grad[i, j, k] for i, j, k in
                         itt.product([x, x + self.sign(dx)], [y, y + self.sign(dy)], [z, z + self.sign(dz)])]
            pos = self.trilinear(neighbors, [abs(dx), abs(dy), abs(dz)])
            dx, dy, dz = -dx, -dy, -dz
            neighbors = [grad[i, j, k] for i, j, k in
                         itt.product([x, x + self.sign(dx)], [y, y + self.sign(dy)], [z, z + self.sign(dz)])]
            neg = self.trilinear(neighbors, [abs(dx), abs(dy), abs(dz)])
            if pos > grad_p or neg > grad_p:
                grad[x, y, z] = 0

        # 先验知识去除干扰
        prior_mask = np.ones((114, 114, 106))
        for x in range(73, 114):
            prior_mask[x, 325 - 3 * x:, :] = 0
        prior_mask[:10, :, :] = 0.1
        prior_mask[-10:, :, :] = 0.1
        prior_mask[:, :10, :] = 0.1
        prior_mask[:, -10:, :] = 0.1
        prior_mask[:, :, :10] = 0.1
        prior_mask[:, :, -10:] = 0.1

        test = prior_mask * mask * (1 - edge) * grad
        test[test < 20] = 0

        # 二次切割
        edge_, mask_, sample_points_ = self.thresholdCut(test, 20)

        pixel_volume = mask_.sum()
        print('估算像素体积：%d' % pixel_volume)
        print(self.spaceInfo(ref_ds))
        vx, vy, vz = self.spaceInfo(ref_ds)
        volume = vx * vy * vz * pixel_volume
        print('体积：%d mm2' % volume)

        mask_ = np.flip(mask_, 1)

        # 检查图片
        # for z in range(0, mask_.shape[2], 10):
        #     print('z: ', z)
        #     img_clip = mask_[:, :, z]
        #     self.showPic(img_clip)
        img_clip = mask_[:, :, 60]
        self.showPic(img_clip)

        return mask_

    def showPic(self, img_clip):
        import matplotlib.pyplot as plt
        plt.figure(dpi=150)
        plt.axes().set_aspect('equal', 'datalim')
        plt.set_cmap(plt.gray())
        plt.imshow(img_clip)
        plt.show()

    def spaceInfo(self, ds):
        space_x, space_y = ds.PixelSpacing
        space_z = ds.SliceThickness
        return float(space_x), float(space_y), float(space_z)

    def gauss(self, k_size, sigma):
        X, Y, Z = [np.linspace(-sigma * 3, sigma * 3, k_size) for _ in range(3)]
        x, y, z = np.meshgrid(X, Y, Z)  # 生成坐标矩阵
        gauss_raw = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2)) / ((2 * np.pi) ** 1.5 * (sigma ** 3))
        return gauss_raw / gauss_raw.sum()

    def conv3d(self, X, Z, padding='valid'):
        import torch
        def pack(arr):
            return torch.tensor(np.array([[arr]]).astype(np.float32))

        return np.array(torch.nn.functional.conv3d(pack(X), pack(Z), padding=padding)[0][0])

    def thresholdLowerCut(self, data, threshold):
        import itertools as itt
        X, Y, Z = data.shape
        edge = np.zeros((X, Y, Z))
        mask = np.zeros((X, Y, Z))
        sample_points = set()
        for y, z in itt.product(range(Y), range(Z)):
            tmp_line = data[:, y, z]
            for x in range(X):
                if tmp_line[x] >= threshold:
                    edge[x, y, z] = 1
                    mask[x:, y, z] = 1
                    sample_points.add((x, y, z))
                    break
        for x, z in itt.product(range(X), range(Z)):
            for y in range(Y):
                if mask[x, y, z]:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    break
            for y in range(Y - 1, 0, -1):
                if mask[x, y, z]:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    break
        for x, y in itt.product(range(X), range(Y)):
            for z in range(Z):
                if mask[x, y, z]:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    break
            for z in range(Z - 1, 0, -1):
                if mask[x, y, z]:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    break
        return edge, mask, sample_points

    def thresholdCut(self, data, threshold):
        import itertools as itt

        X, Y, Z = data.shape
        edge = np.zeros((X, Y, Z))
        mask = np.zeros((X, Y, Z))
        sample_points = set()
        for y, z in itt.product(range(Y), range(Z)):
            tmp_line = data[:, y, z]
            x1, x2 = 1919810, 114514
            for x in range(X):
                if tmp_line[x] >= threshold:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    x1 = x
                    break
            for x in range(X - 1, 0, -1):
                if tmp_line[x] >= threshold:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    x2 = x
                    break
            if x1 <= x2:
                mask[x1:x2, y, z] = 1
        for x, z in itt.product(range(X), range(Z)):
            tmp_line = data[x, :, z]
            y1, y2 = 1919810, 114514
            for y in range(Y):
                if tmp_line[y] >= threshold:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    y1 = y
                    break
            for y in range(Y - 1, 0, -1):
                if tmp_line[y] >= threshold:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    y2 = y
                    break
            if y1 <= y2:
                mask[x, y1:y2, z] = 1
        for x, y in itt.product(range(X), range(Z)):
            tmp_line = data[x, y, :]
            z1, z2 = 1919810, 114514
            for z in range(Z):
                if tmp_line[z] >= threshold:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    z1 = z
                    break
            for z in range(Z - 1, 0, -1):
                if tmp_line[z] >= threshold:
                    edge[x, y, z] = 1
                    sample_points.add((x, y, z))
                    z2 = z
                    break
            if z1 <= z2:
                mask[x, y, z1:z2] = 1
        return edge, mask, sample_points

    # 非极大值抑制
    def trilinear(self, neighbors, gradient):
        dx, dy, dz = gradient
        vx = [(1 - dx) * neighbors[i] + dx * neighbors[i + 4] for i in range(4)]
        vy = [(1 - dy) * vx[i] + dy * vx[i + 2] for i in range(2)]
        vz = (1 - dz) * vy[0] + dz * vy[1]
        return vz

    def sign(self, x):
        return 1 if x >= 0 else -1


#
# BreastVolumeCalculatorTest
#

class BreastVolumeCalculatorTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
    """
        self.setUp()
        # self.test_BreastVolumeCalculator1()

    def test_BreastVolumeCalculator1(self):
        self.delayDisplay("Starting the test")
        import numpy.testing as npt
        self.delayDisplay('Test passed')
