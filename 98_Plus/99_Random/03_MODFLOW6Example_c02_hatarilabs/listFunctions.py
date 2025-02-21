import math
import numpy as np
from scipy.interpolate import griddata

class listFunctions:
    def __init__(self, name):
        self.name = name
            
    def listWaterTableCellFunction(modDis,modHds):
        #empty numpy array for the water table
        waterTableCellGrid = np.zeros((modDis['cellRows'],modDis['cellCols']))

        #obtain the first positive or real head from the head array
        for row in range(modDis['cellRows']):
            for col in range(modDis['cellCols']):
                anyList = []
                for lay in range(modDis['cellLays']):
                    anyList.append(modHds['cellHeadGrid']['lay' + str(lay)][row,col])
                a = np.asarray(anyList)
                if list(a[a>-1e+10]) != []:                    #just in case there are some inactive zones
                    waterTableCellGrid[row,col] = a[a>-1e+10][0]
                else: waterTableCellGrid[row,col] = -1e+10
                
        return list(waterTableCellGrid.flatten())

    def listWaterTableVertexFunction(modDis,modHds):
        #empty numpy array for the water table
        waterTableVertexGrid = np.zeros((modDis['vertexRows'],modDis['vertexCols']))

        #obtain the first positive or real head from the head array
        for row in range(modDis['vertexRows']):
            for col in range(modDis['vertexCols']):
                anyList = []
                for lay in range(modDis['cellLays']):
                    anyList.append(modHds['vertexHeadGrid']['lay' + str(lay)][row,col])
                a = np.asarray(anyList)
                if list(a[a>-1e+10]) != []:                    #just in case there are some inactive zones
                    waterTableVertexGrid[row,col] = a[a>-1e+10][0]
                else: waterTableVertexGrid[row,col] = -1e+10
        return list(waterTableVertexGrid.flatten())
    
##########################################    
# Lists for the VTK file of Model Geometry  
##########################################  
    
    def vertexXYZPointsFunction(modDis):
        #empty list to store all vertex XYZ
        vertexXYZPoints = []

        #definition of xyz points for all vertex
        for lay in range(modDis['vertexLays']):
            for row in range(modDis['vertexRows']):
                for col in range(modDis['vertexCols']):
                    xyz=[
                        modDis['vertexEastingArray1D'][col], 
                        modDis['vertexNorthingArray1D'][row],
                        modDis['vertexZGrid']['lay'+str(lay)][row, col]
                        ]
                    vertexXYZPoints.append(xyz)
                    
        return vertexXYZPoints
    
    def vertexWaterTableXYZPointsFunction(listWaterTableVertex,modDis):
        #empty list to store all vertex Water Table XYZ
        vertexWaterTableXYZPoints = []
        
        gridWaterTableVertex = np.array(listWaterTableVertex).reshape(modDis["vertexRows"],modDis["vertexCols"])

        #definition of xyz points for all vertex
        for row in range(modDis['vertexRows']):
            for col in range(modDis['vertexCols']):
                if gridWaterTableVertex[row, col] > -1e+10:
                    waterTable = gridWaterTableVertex[row, col]
                else:
                    waterTable = 1000
                xyz=[
                    modDis['vertexEastingArray1D'][col], 
                    modDis['vertexNorthingArray1D'][row],
                    waterTable
                    ]
                vertexWaterTableXYZPoints.append(xyz)
                
        return vertexWaterTableXYZPoints
    
    def listCellHeadsFunction(lays,grid,modDis,modHds):
        #empty list to store all cell heads
        anyList = []

        #definition of cellHead
        for lay in range(modDis[lays]):
            for item in list(modHds[grid]['lay'+str(lay)].flatten()):
                anyList.append(item)
                
        return anyList
       
    def bcCellsListFunction(modFile,keyName,listHexaSequence,modDis):
        #definition of cells
        anyGrid = np.zeros((modDis['cellLays'],modDis['cellRows'],modDis['cellCols']))
        
        for item in modFile[keyName]:
            anyGrid[item[0]-1,item[1]-1,item[2]-1] = 1
            
        listBcCellsIO = list(anyGrid.flatten())
        
        listBcCellsIODef = []
        listBcCellsSecuenceDef = []
        
        for item in range(len(listBcCellsIO)):
            if listBcCellsIO[item] > 0:
                listBcCellsIODef.append(listBcCellsIO[item])
                listBcCellsSecuenceDef.append(listHexaSequence[item])
        
        
        return [listBcCellsIODef,listBcCellsSecuenceDef]
    
##################################################   
# Hexahedrons and Quads sequences for the VTK File
##################################################  

    def listLayerQuadSequenceFunction(modDis):
        anyQuadList = []
        
        #definition of hexahedrons cell coordinates
        for row in range(modDis['cellRows']):
            for col in range(modDis['cellCols']):
                pt0 = modDis['vertexCols']*(row+1)+col
                pt1 = modDis['vertexCols']*(row+1)+col+1
                pt2 = modDis['vertexCols']*(row)+col+1
                pt3 = modDis['vertexCols']*(row)+col
                anyList = [pt0,pt1,pt2,pt3]
                anyQuadList.append(anyList)
                
        return anyQuadList
    
    def listHexaSequenceFunction(modDis):
        #empty list to store cell coordinates
        listHexaSequence = []

        #definition of hexahedrons cell coordinates
        for lay in range(modDis['cellLays']):
            for row in range(modDis['cellRows']):
                for col in range(modDis['cellCols']):
                    pt0 = modDis['vertexPerLay']*(lay+1)+modDis['vertexCols']*(row+1)+col
                    pt1 = modDis['vertexPerLay']*(lay+1)+modDis['vertexCols']*(row+1)+col+1
                    pt2 = modDis['vertexPerLay']*(lay+1)+modDis['vertexCols']*(row)+col+1
                    pt3 = modDis['vertexPerLay']*(lay+1)+modDis['vertexCols']*(row)+col
                    pt4 = modDis['vertexPerLay']*(lay)+modDis['vertexCols']*(row+1)+col
                    pt5 = modDis['vertexPerLay']*(lay)+modDis['vertexCols']*(row+1)+col+1
                    pt6 = modDis['vertexPerLay']*(lay)+modDis['vertexCols']*(row)+col+1
                    pt7 = modDis['vertexPerLay']*(lay)+modDis['vertexCols']*(row)+col
                    anyList = [pt0,pt1,pt2,pt3,pt4,pt5,pt6,pt7]
                    listHexaSequence.append(anyList)
                    
        return listHexaSequence
    
    