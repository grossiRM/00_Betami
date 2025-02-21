import math
import numpy as np
from scipy.interpolate import griddata

class transFunctions:
    def __init__(self, name):
        self.name = name
        
    #function that return a dictionary of z values on the vertex
    def interpolateCelltoVertex(modDis,item):
        dictZVertex = {}
        #for lay in modDis[item].keys():
        #    values = np.asarray(modDis[item][lay])
        #    grid_z = griddata(modDis['cellCentroids'], values, 
        #                  (modDis['vertexXGrid'], modDis['vertexYGrid']), 
        #                  method='nearest')
        #    dictZVertex[lay]=grid_z
        
        #arrange to hace positive heads in all vertex of an active cell
        for lay in range(len(modDis[item].keys())):
            
            matrix = np.zeros([modDis['vertexRows'],modDis['vertexCols']])
            
            matrix[0,:-1] = modDis[item]['lay'+str(lay)][0,:]
            matrix[:-1,0] = modDis[item]['lay'+str(lay)][:,0]
            
            for row in range(1,modDis['cellRows']):
                for col in range(1,modDis['cellCols']):
                    headLay = modDis[item]['lay'+str(lay)]
                    neighcartesianlist = [headLay[row-1,col-1],headLay[row-1,col],headLay[row,col-1],headLay[row,col]]

                    headMean = sum(neighcartesianlist)/len(neighcartesianlist)
            
                    matrix[row,col]=headMean
            
            matrix[-1,:-1] = modDis[item]['lay'+str(lay)][-1,:]
            matrix[:-1,-1] = modDis[item]['lay'+str(lay)][:,-1]
            matrix[-1,-1] = modDis[item]['lay'+str(lay)][-1,-1]

            dictZVertex['lay'+str(lay)]=matrix
            
        return dictZVertex
        
    def vertexHeadGridCentroidFunction(modDis,modHds):
        
        vertexHeadGridCentroid = {}
        
        #arrange to hace positive heads in all vertex of an active cell
        for lay in range(modDis['cellLays']):
            matrix = np.zeros([modDis['vertexRows'],modDis['vertexCols']])
            
            matrix[0,:-1] = modHds['cellHeadGrid']['lay'+str(lay)][0,:]
            matrix[:-1,0] = modHds['cellHeadGrid']['lay'+str(lay)][:,0]
            
            for row in range(1,modDis['cellRows']):
                for col in range(1,modDis['cellCols']):
                    headLay = modHds['cellHeadGrid']['lay'+str(lay)]
                    neighcartesianlist = [headLay[row-1,col-1],headLay[row-1,col],headLay[row,col-1],headLay[row,col]]
                    headList = []
                    for item in neighcartesianlist:
                        if item > -1e+20:
                            headList.append(item)
                    if len(headList) > 0:
                        headMean = sum(headList)/len(headList)
                    else:
                        headMean = -1e+20
            
                    matrix[row,col]=headMean
                
            matrix[-1,:-1] = modHds['cellHeadGrid']['lay'+str(lay)][-1,:]
            matrix[:-1,-1] = modHds['cellHeadGrid']['lay'+str(lay)][:,-1]
            matrix[-1,-1] = modHds['cellHeadGrid']['lay'+str(lay)][-1,-1]

            vertexHeadGridCentroid['lay'+str(lay)]=matrix
            
        return vertexHeadGridCentroid
    
    def vertexHeadGridFunction(vertexHeadGridCentroid,modDis,modHds):
        
        vertexHeadGrid = {}
        
        for lay in range(modDis['vertexLays']):
            anyGrid = vertexHeadGridCentroid
            if lay == modDis['cellLays']:
                vertexHeadGrid['lay'+str(lay)] = anyGrid['lay'+str(lay-1)]
            elif lay == 0:
                vertexHeadGrid['lay0'] = anyGrid['lay0']
            else:
                value = np.where(anyGrid['lay'+str(lay)]>-1e+20,
                                 anyGrid['lay'+str(lay)],
                                 (anyGrid['lay'+str(lay-1)] + anyGrid['lay'+str(lay)])/2
                                  )
                vertexHeadGrid['lay'+str(lay)] = value
                
        return vertexHeadGrid

