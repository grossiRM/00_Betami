
import numpy as np

class vtkFunctions:
    def __init__(self, name):
        self.name = name
        
    def printHeader(vtkText,pointNumber,cellNumber):
        #add header
        vtkText.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        vtkText.write('  <UnstructuredGrid>\n')
        vtkText.write('    <Piece NumberOfPoints="'+str(pointNumber)+'" NumberOfCells="'+str(cellNumber)+'">\n')
        
    def printPointData(vtkText,pointDataName,pointDataList):
        #point data
        vtkText.write('      <PointData Scalars="Model">\n')  #Scalars="Heads"
        vtkText.write('        <DataArray type="Float64" Name="' + pointDataName + '" format="ascii">\n')
        for item  in range(len(pointDataList)):
            textvalue = str(pointDataList[item])
            if item == 0:
                vtkText.write('          ' + textvalue + ' ')
            elif item % 20 == 0:
                vtkText.write(textvalue + '\n          ')
            else:
                vtkText.write(textvalue + ' ')
        vtkText.write('\n')
        vtkText.write('        </DataArray>\n')
        vtkText.write('      </PointData>\n')
        
    def printCellData(vtkText,cellDataName,cellDataList):
        #cell data
        vtkText.write('      <CellData Scalars="Model" >\n') #Scalars="Model"
        vtkText.write('        <DataArray type="Float64" Name="' + cellDataName + '" format="ascii">\n')
        for item  in range(len(cellDataList)): #cell list
            textvalue = str(cellDataList[item])
            if item == 0:
                vtkText.write('          ' + textvalue + ' ')
            elif item % 20 == 0:
                vtkText.write(textvalue + '\n          ')
            else:
                vtkText.write(textvalue + ' ')
        vtkText.write('\n')
        vtkText.write('        </DataArray>\n')
        vtkText.write('      </CellData>\n')
        
    def printPointDefinition(vtkText,vertexXYZPoints):
        #points definition
        vtkText.write('      <Points>\n')
        vtkText.write('        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">\n')
        for item in range(len(vertexXYZPoints)):
            tuplevalue = tuple(vertexXYZPoints[item])
            if item == 0:
                vtkText.write("          %.2f %.2f %.2f " % tuplevalue)
            elif item % 4 == 0:        
                vtkText.write("%.2f %.2f %.2f \n          " % tuplevalue)
            elif item == len(vertexXYZPoints)-1:
                vtkText.write("%.2f %.2f %.2f \n" % tuplevalue)
            else:
                vtkText.write("%.2f %.2f %.2f " % tuplevalue)   
        vtkText.write('        </DataArray>\n')        
        vtkText.write('      </Points>\n')
        
    def printCellQuadConnectivityOffsetType(vtkText,listQuadSequence):
        #cell connectivity
        vtkText.write('      <Cells>\n')               
        vtkText.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for item  in range(len(listQuadSequence)):
            vtkText.write('          ')
            vtkText.write('%s %s %s %s \n' % tuple(listQuadSequence[item]))
        vtkText.write('        </DataArray>\n') 
        #cell offsets
        vtkText.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        for item in range(len(listQuadSequence)):
            offset = str((item+1)*4)
            if item == 0:
                vtkText.write('          ' + offset + ' ')
            elif item % 20 == 0:
                vtkText.write(offset + ' \n          ')
            elif item == len(listQuadSequence)-1:
                vtkText.write(offset + ' \n')
            else:
                vtkText.write(offset + ' ')
        vtkText.write('        </DataArray>\n') 
        #cell types
        vtkText.write('        <DataArray type="UInt8" Name="types" format="ascii">\n') 
        for item in range(len(listQuadSequence)):
            if item == 0:
                vtkText.write('          ' + '9 ')
            elif item % 20 == 0:
                vtkText.write('9 \n          ')
            elif item == len(listQuadSequence)-1:
                vtkText.write('9 \n')
            else:
                vtkText.write('9 ')
        vtkText.write('        </DataArray>\n')
        vtkText.write('      </Cells>\n')  
        
    def printCellHexaConnectivityOffsetType(vtkText,listHexaSequence):
        #cell connectivity
        vtkText.write('      <Cells>\n')               
        vtkText.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for item  in range(len(listHexaSequence)):
            vtkText.write('          ')
            vtkText.write('%s %s %s %s %s %s %s %s \n' % tuple(listHexaSequence[item]))
        vtkText.write('        </DataArray>\n') 
        #cell offset
        vtkText.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        for item in range(len(listHexaSequence)):
            offset = str((item+1)*8)
            if item == 0:
                vtkText.write('          ' + offset + ' ')
            elif item % 20 == 0:
                vtkText.write(offset + ' \n          ')
            elif item == len(listHexaSequence)-1:
                vtkText.write(offset + ' \n')
            else:
                vtkText.write(offset + ' ')
        vtkText.write('        </DataArray>\n') 
        #cell type
        vtkText.write('        <DataArray type="UInt8" Name="types" format="ascii">\n') 
        for item in range(len(listHexaSequence)):
            if item == 0:
                vtkText.write('          ' + '12 ')
            elif item % 20 == 0:
                vtkText.write('12 \n          ')
            elif item == len(listHexaSequence)-1:
                vtkText.write('12 \n')
            else:
                vtkText.write('12 ')        
        vtkText.write('        </DataArray>\n')
        vtkText.write('      </Cells>\n') 
    
    def printFooter(vtkText):
        vtkText.write('    </Piece>\n')
        vtkText.write('  </UnstructuredGrid>\n')
        vtkText.write('</VTKFile>\n')
        
        
        
        
        
        