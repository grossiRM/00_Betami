import math
import numpy as np
from scipy.interpolate import griddata

class workFunctions:
    def __init__(self, name):
        self.name = name
        
    def getListFromDel(delta,modDis,disLines,itemsperrow=10):
        if delta == 'delr':
            cellDim = modDis['cellCols']
        elif delta == 'delc':
            cellDim = modDis['cellRows']
        else: pass
    
        for line in disLines:
            if delta in line:
                indexDel = disLines.index(line) + 1
            else:
                pass
    
        if 'CONSTANT' in disLines[indexDel]:
            constElevation = float(disLines[indexDel].split()[1])
            anyLines = [constElevation for x in range(cellDim)]
        
        elif 'INTERNAL' in disLines[indexDel]:
            #empty array and number of lines 
            anyLines = []
            #final breaker
            finalbreaker = indexDel+1+math.ceil(celldim/itemsperrow)
            #append to list all items
            for line in range(indexDel+1,finalbreaker,1):
                listItems = [float(item) for item in disLines[line].split()]
                for item in listItems: anyLines.append(item)
        else:
            anylines = []
        return np.asarray(anyLines)  
    
    
    def getUniLayerListFromTerm(modDis,fileLines,term,itemperrow=10):    
    
        for line in fileLines:
            if term in line:
                indexItem = fileLines.index(line) + 1
            else:
                pass
    
        if 'CONSTANT' in fileLines[indexItem]:
            constElevation = float(fileLines[indexItem].split()[1])
            anyLines = [constElevation for x in range(modDis['cellsPerLay'])]

        else:
            anylines = []
        return np.asarray(anyLines)  
    
    def getListinDictxLayFromGriddataLayered(modFile,fileLines,cellType,modDis):    
   
        for line in fileLines:
            if cellType in line:
                indexItem = fileLines.index(line) + 1
            else: pass
            
        #read all layers
        anyLayers = []
        
        while len(anyLayers) <= modDis['cellLays']:
            for line in range(indexItem,len(fileLines)):
                if 'CONSTANT' in fileLines[line]:
                    constElevation = float(fileLines[line].split()[1])
                    anyLines = [constElevation for x in range(modDis['cellsPerLay'])]
                    anyLayers.append(anyLines)
                else:pass
        
        #build dict
        anyDict = {}
        
        i = 0
        for lay in range(modDis['cellLays']):
            anyDict['lay'+str(lay)]=anyLayers[lay]
            i+=1
        
        return anyDict
    
    def getTermFromKeyword(fileLines,term,keyword):      
        for line in fileLines:
            if term in line:
                return int(line.split()[1])
            else:
                pass
            
    def getCellsforBoundary(fileLines,bcType,maxbound,period):
        
        indexList = 0 
        for line in fileLines:
            if 'BEGIN period  '+str(period) in line:
                indexList = fileLines.index(line) + 1
        
        anyList = []
        
        if bcType in ['drn']:
            for item in range(indexList, indexList+maxbound):
                anyLine = fileLines[item].split()
                anyList.append([ int(anyLine[0]), int(anyLine[1]), int(anyLine[2]), float(anyLine[3]), float(anyLine[4])])
        elif bcType in ['chd','wel']:
            for item in range(indexList, indexList+maxbound):
                anyLine = fileLines[item].split()
                anyList.append([ int(anyLine[0]), int(anyLine[1]), int(anyLine[2]), float(anyLine[3])])
        else:
            pass
                               
        return anyList
            
    

