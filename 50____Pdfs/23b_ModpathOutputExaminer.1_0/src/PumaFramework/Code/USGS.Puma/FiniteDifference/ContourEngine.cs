using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;
using USGS.Puma.NTS.Geometries;
using GeoAPI.Geometries;

namespace USGS.Puma.FiniteDifference
{
    /// <summary>
    /// Provides contouring functionality for layer data associated with an areal
    /// finite-difference grid.
    /// </summary>
    /// <remarks></remarks>
    public class ContourEngine
    {
        #region Fields
        /// <summary>
        /// 
        /// </summary>
        private CellCenteredArealGrid _RGrid;
        /// <summary>
        /// 
        /// </summary>
        private bool[,] _HOut;
        /// <summary>
        /// 
        /// </summary>
        private bool[,] _VOut;
        /// <summary>
        /// 
        /// </summary>
        private bool[,] _HBranch;
        /// <summary>
        /// 
        /// </summary>
        private bool[,] _VBranch;
        /// <summary>
        /// 
        /// </summary>
        private float[] _CI = null;
        #endregion

        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="ContourEngine"/> class.
        /// </summary>
        /// <remarks></remarks>
        public ContourEngine()
        {
            Grid = null;
            _ExcludedValues = new List<float>();

            _CI = new float[20];
            _CI[0] = 0.01f;
            _CI[1] = 0.02f;
            _CI[2] = 0.05f;
            _CI[3] = 0.1f;
            _CI[4] = 0.2f;
            _CI[5] = 0.5f;
            _CI[6] = 1f;
            _CI[7] = 2f;
            _CI[8] = 5f;
            _CI[9] = 10f;
            _CI[10] = 20f;
            _CI[11] = 50f;
            _CI[12] = 100f;
            _CI[13] = 200f;
            _CI[14] = 500f;
            _CI[15] = 1000f;
            _CI[16] = 2000f;
            _CI[17] = 5000f;
            _CI[18] = 10000f;
            _CI[19] = 20000f;

            UseDefaultNoDataRange = false;
            NoDataRangeMinimum = 1000000f;
            MaximumContourCount = 250;

        }
        /// <summary>
        /// Initializes a new instance of the <see cref="ContourEngine"/> class.
        /// </summary>
        /// <param name="grid">The grid.</param>
        /// <remarks></remarks>
        public ContourEngine(CellCenteredArealGrid grid) : this()
        { 
            Grid = grid;
        }
        #endregion

        #region Public Members
        /// <summary>
        /// 
        /// </summary>
        private CellCenteredArealGrid _Grid;
        /// <summary>
        /// Areal Finite-difference Grid
        /// </summary>
        /// <value>The grid.</value>
        /// <remarks></remarks>
        public CellCenteredArealGrid Grid
        {
            get { return _Grid; }
            set 
            { 
                if (value != null)
                {
                    if (LayerArray != null)
                    {
                        if (value.RowCount != LayerArray.RowCount || value.ColumnCount != LayerArray.ColumnCount)
                            throw new ArgumentException("Grid dimensions do not match current layer array.");
                    }
                    _Grid = value;
                }
                else
                {
                    _Grid = null;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        private Array2d<float> _LayerArray;
        /// <summary>
        /// Gets or sets the layer array.
        /// </summary>
        /// <value>The layer array.</value>
        /// <remarks></remarks>
        public Array2d<float> LayerArray
        {
            get { return _LayerArray; }
            set 
                {
                    if (Grid != null)
                    {
                        if (value.RowCount != Grid.RowCount || value.ColumnCount != Grid.ColumnCount)
                            throw new ArgumentException("Array dimensions do not match the current grid.");
                    }
                    _LayerArray = value; 
                }
        }

        /// <summary>
        /// 
        /// </summary>
        private bool _UseDefaultNoDataRange;
        /// <summary>
        /// Gets or sets a value indicating whether to use the default range for no-data values.
        /// </summary>
        /// <value><c>true</c> to use the default no-data range; otherwise, <c>false</c>.</value>
        /// <remarks>
        /// If this flag is <c>true</c>, a all cell values greater than or equal to a specified 
        /// value are considered to be no-data values that are omitted from the contoured region.
        /// </remarks>
        public bool UseDefaultNoDataRange
        {
            get { return _UseDefaultNoDataRange; }
            set { _UseDefaultNoDataRange = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private float _NoDataRangeMinimum;
        /// <summary>
        /// Gets or sets the no data range minimum.
        /// </summary>
        /// <value>The no-data range minimum.</value>
        /// <remarks>
        /// When <see cref="P:ContourEngine.UseNoDataRange"></see> is <c>true</c>, all cell values greater than or equal to
        /// this value are considered to be no-data values that are omitted from the contoured region.
        /// </remarks>
        public float NoDataRangeMinimum
        {
            get { return _NoDataRangeMinimum; }
            set { _NoDataRangeMinimum = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private List<float> _ExcludedValues;
        /// <summary>
        /// Gets or sets the no data values.
        /// </summary>
        /// <value>A collection of no-data values.</value>
        /// <remarks></remarks>
        public List<float> ExcludedValues
        {
            get 
            {
                if (_ExcludedValues == null)
                    _ExcludedValues = new List<float>();
                return _ExcludedValues; 
            }
            set { _ExcludedValues = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _MaximumContourCount;
        /// <summary>
        /// Gets or sets the maximum contour count.
        /// </summary>
        /// <value>The maximum contour count.</value>
        /// <remarks></remarks>
        public int MaximumContourCount
        {
            get { return _MaximumContourCount; }
            set { _MaximumContourCount = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private List<float> _ContourLevels;
        /// <summary>
        /// Gets or sets the contour levels.
        /// </summary>
        /// <value>The contour levels.</value>
        /// <remarks></remarks>
        public List<float> ContourLevels
        {
            get { return _ContourLevels; }
            set { _ContourLevels = value; }
        }

        /// <summary>
        /// Generates the constant intervals.
        /// </summary>
        /// <param name="contourInterval">The contour interval.</param>
        /// <param name="referenceContour">The reference contour.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public List<float> GenerateConstantIntervals(float contourInterval, float referenceContour)
        {
            if (LayerArray == null)
                throw new Exception("LayerArray is not set.");

            return GenerateConstantIntervals(contourInterval, referenceContour, float.MinValue, float.MaxValue);

        }
        /// <summary>
        /// Generates the constant intervals.
        /// </summary>
        /// <param name="contourInterval">The contour interval.</param>
        /// <param name="referenceContour">The reference contour.</param>
        /// <param name="minimumLevel">The minimum level.</param>
        /// <param name="maximumLevel">The maximum level.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public List<float> GenerateConstantIntervals(float contourInterval, float referenceContour, float minimumLevel, float maximumLevel)
        {
            List<float> contourIntervals = null;
            float minValue;
            float maxValue;
            float contourLevel;

            if (LayerArray == null)
                throw new Exception("LayerArray is not set.");

            contourIntervals = new List<float>();

            ArrayUtility aUtil = new ArrayUtility();
            //if (NoDataValues != null && NoDataValues.Count > 0)
            //    aUtil.FindMinimumAndMaximum(LayerArray, out minValue, out maxValue, NoDataValues.ToArray());
            //else
            //    aUtil.FindMinimumAndMaximum(LayerArray, out minValue, out maxValue);

            ArrayStatistics<float> aStats = GetMinAndMaxValues();
            if (aStats == null) return null;
            minValue = aStats.MinimumValue;
            maxValue = aStats.MaximumValue;
            
            if (minimumLevel > minValue) minValue = minimumLevel;
            if (maximumLevel < maxValue) maxValue = maximumLevel;

            // check the reference contour and add it to the list if it is within the range
            if (referenceContour >= minValue && referenceContour <= maxValue)
                contourIntervals.Add(referenceContour);

            // next work forward from the reference contour and find all of the contours in the
            // range.
            contourLevel = referenceContour + contourInterval;
            while (contourLevel <= maxValue)
            {
                if (contourLevel >= minValue) contourIntervals.Add(contourLevel);
                contourLevel += contourInterval;
                if (contourIntervals.Count > MaximumContourCount) contourLevel = float.MaxValue;
            }

            // now work backward from the reference contour and find the rest of the contours
            // within the range.
            if (contourIntervals.Count < MaximumContourCount)
            {
                contourLevel = referenceContour - contourInterval;
                while (contourLevel >= minValue)
                {
                    if (contourLevel <= maxValue) contourIntervals.Add(contourLevel);
                    contourLevel -= contourInterval;
                }
            }

            // sort the values in ascending order
            contourIntervals.Sort();

            return contourIntervals;
            

        }

        /// <summary>
        /// Creates the contours.
        /// </summary>
        /// <param name="contourLevels">The contour levels.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ContourLineList CreateContours(List<float> contourLevels)
        {
            if (LayerArray == null)
                throw new Exception("No layer array is specified.");
            if (contourLevels == null)
                throw new NullReferenceException();
            if (contourLevels.Count == 0)
                throw new Exception("The contour levels list is empty.");

            return CreateContours(contourLevels, 1, 1, LayerArray.RowCount, LayerArray.ColumnCount);
        
        }
        /// <summary>
        /// Creates the contours.
        /// </summary>
        /// <param name="contourLevels">The contour levels.</param>
        /// <param name="firstRow">The first row.</param>
        /// <param name="firstColumn">The first column.</param>
        /// <param name="lastRow">The last row.</param>
        /// <param name="lastColumn">The last column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ContourLineList CreateContours(List<float> contourLevels, int firstRow,int firstColumn,int lastRow,int lastColumn)
        {
            if (LayerArray == null)
                throw new Exception("No layer array is specified.");

            if (contourLevels == null)
                throw new NullReferenceException("contourLevels list is not set.");
            
            if ( !CheckRowColumnRange(firstRow,firstColumn,lastRow,lastColumn) )
                throw new ArgumentException("The specified row and column range is not valid.");
            
            // If no grid has been defined, create a new grid with the row and layer
            // dimensions of the layer array, origin = (0,0), square 1 x 1 cells, and rotation angle = 0.
            // If a grid has been defined, use that grid in the calculations.
            if (Grid == null)
                _RGrid = new CellCenteredArealGrid(LayerArray.RowCount, LayerArray.ColumnCount, 1, (ICoordinate)(new Point(0, 0)), 0);
            else
                _RGrid = Grid;

            // Create branch arrays
            _HOut = new bool[_RGrid.RowCount, _RGrid.ColumnCount];
            _VOut = new bool[_RGrid.RowCount, _RGrid.ColumnCount];
            _HBranch = new bool[_RGrid.RowCount, _RGrid.ColumnCount];
            _VBranch = new bool[_RGrid.RowCount, _RGrid.ColumnCount];

            ContourLineList contours = new ContourLineList();
            ContourLineList list;

            // Loop through all contour levels
            for (int i = 0; i < contourLevels.Count; i++)
            {
                list = ProcessContourLevel(contourLevels[i], firstRow, firstColumn, lastRow, lastColumn);
                foreach (ContourLine item in list)
                { contours.Add(item); }
            }

            // Delete branch arrays
            _HOut = null;
            _VOut = null;
            _HBranch = null;
            _VBranch = null;

            // Return result
            return contours;

        }
        /// <summary>
        /// Creates the contours.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public ContourLineList CreateContours()
        {
            return CreateContours(this.ContourLevels);
        }
        /// <summary>
        /// Creates the contours.
        /// </summary>
        /// <param name="firstRow">The first row.</param>
        /// <param name="firstColumn">The first column.</param>
        /// <param name="lastRow">The last row.</param>
        /// <param name="lastColumn">The last column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public ContourLineList CreateContours(int firstRow, int firstColumn, int lastRow, int lastColumn)
        {
            return CreateContours(this.ContourLevels, firstRow, firstColumn, lastRow, lastColumn);
        }

        /// <summary>
        /// Gets the min and max values.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public ArrayStatistics<float> GetMinAndMaxValues()
        {
            float[] noDataArray = null;
            if (LayerArray == null) return null;

            IArrayUtility<float> aUtil = new ArrayUtility();

            if (UseDefaultNoDataRange)
            {
                Array2d<float> a = new Array2d<float>(LayerArray);
                noDataArray = new float[1];
                float ndv = 1.0E+30f;
                noDataArray[0] = ndv;
                float noDataVal = NoDataRangeMinimum;
                for (int row = 1; row <= LayerArray.RowCount; row++)
                {
                    for (int col = 1; col <= LayerArray.ColumnCount; col++)
                    {
                        if (a[row, col] > noDataVal) a[row, col] = ndv ;
                    }
                }
                return aUtil.FindMinimumAndMaximum(a, noDataArray);
            }
            else
            {
                if (ExcludedValues != null && ExcludedValues.Count > 0) noDataArray = ExcludedValues.ToArray();
                return aUtil.FindMinimumAndMaximum(LayerArray, noDataArray);
            }


        }

        /// <summary>
        /// Computes the contour interval.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public float ComputeContourInterval()
        {
            try
            {
                ArrayStatistics<float> aStats = GetMinAndMaxValues();
                float range = aStats.MaximumValue - aStats.MinimumValue;
                float numCon = 0f;
                float target = 25f;

                for (int n = 0; n < _CI.Length; n++)
                {
                    numCon = range / _CI[n];
                    if ( numCon <= target)
                    {
                        return _CI[n];
                    }
                }

                return 1;

            }
            catch (Exception)
            {
                return 1;
            }
        }

        #endregion

        #region Private Members
        /// <summary>
        /// Initializes the branch flags.
        /// </summary>
        /// <remarks></remarks>
        private void InitializeBranchFlags()
        {
            // reset all values
            for (int i = 0; i < LayerArray.RowCount; i++)
            {
                for (int j = 0; j < LayerArray.ColumnCount; j++)
                {
                    if (j == LayerArray.ColumnCount-1)
                        _HOut[i, j] = true;
                    else
                        _HOut[i, j] = false;

                    if (i == LayerArray.RowCount - 1)
                        _VOut[i, j] = true;
                    else
                        _VOut[i, j] = false;
                }
            }

            // process no data values
            for (int i = 0; i < LayerArray.RowCount; i++)
            {
                for (int j = 0; j < LayerArray.ColumnCount; j++)
                {
                    if (IsNoDataValue(LayerArray[i+1, j+1]))
                    {
                        _HOut[i, j] = true;
                        _VOut[i, j] = true;
                        if (i > 0)
                        {
                            _VOut[i - 1, j] = true;
                        }
                        if (j > 0)
                        {
                            _HOut[i, j - 1] = true;
                        }
                    }
                }
            }

            // initialize the scratch array flags
            for (int i = 0; i < LayerArray.RowCount; i++)
            {
                for (int j = 0; j < LayerArray.ColumnCount; j++)
                {
                    _HBranch[i, j] = _HOut[i, j];
                    _VBranch[i, j] = _VOut[i, j];
                }
            }


        }

        /// <summary>
        /// Determines whether the specified data value contains level.
        /// </summary>
        /// <param name="dataValue">The data value.</param>
        /// <param name="nodeValue1">The node value1.</param>
        /// <param name="nodeValue2">The node value2.</param>
        /// <returns><c>true</c> if the specified data value contains level; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        private bool ContainsLevel(float dataValue, float nodeValue1,float nodeValue2)
        {
            if (nodeValue2 >= nodeValue1)
                return (dataValue >= nodeValue1 && dataValue <= nodeValue2);
            else
                return (dataValue >= nodeValue2 && dataValue <= nodeValue1);
        }

        /// <summary>
        /// Checks the row column range.
        /// </summary>
        /// <param name="firstRow">The first row.</param>
        /// <param name="firstColumn">The first column.</param>
        /// <param name="lastRow">The last row.</param>
        /// <param name="lastColumn">The last column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private bool CheckRowColumnRange(int firstRow, int firstColumn, int lastRow, int lastColumn)
        {
            bool validRange = true;
            if (firstRow > lastRow || firstColumn > lastColumn) validRange = false;
            if (firstRow < 1 || firstRow > LayerArray.RowCount) validRange = false;
            if (lastRow < 1 || lastRow > LayerArray.RowCount) validRange = false;
            if (firstColumn < 1 || firstColumn > LayerArray.ColumnCount) validRange = false;
            if (lastColumn < 1 || lastColumn > LayerArray.ColumnCount) validRange = false;
            return validRange;
        }

        /// <summary>
        /// Finds the coord point.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="row">The row.</param>
        /// <param name="column">The column.</param>
        /// <param name="branch">The branch.</param>
        /// <param name="level">The level.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private ICoordinate FindCoordPoint(Array2d<float> buffer, int row, int column, int branch, float level)
        {
            if (buffer == null) 
                throw new NullReferenceException();

            try
            {
                float w = 0;
                ICoordinate pt = null;
                ICoordinate pt1 = null;
                ICoordinate pt2 = null;
                double rx = 0;
                double ry = 0;

                switch (branch)
                {
                    case 1:
                        // Find point on horizontal branch
                        if (column > 0 && column < buffer.ColumnCount)
                        {
                            w = (level - buffer[row, column]) / (buffer[row, column + 1] - buffer[row, column]);
                            if (w >= 0F && w <= 1F)
                            {
                                pt1 = _RGrid.GetNodePointRelativeToGrid(new GridCell(row, column));
                                pt2 = _RGrid.GetNodePointRelativeToGrid(new GridCell(row, column + 1));
                                ry = pt1.Y;
                                rx = (1 - w) * pt1.X + w * pt2.X;
                                pt = (ICoordinate)(new Coordinate(rx, ry));
                                _RGrid.TransformRelativeToGlobal(pt);
                            }
                        }
                        break;

                    case 2:
                        // Find point on vertical branch
                        if (row > 0 && row < buffer.RowCount)
                        {
                            w = (level - buffer[row, column]) / (buffer[row + 1, column] - buffer[row, column]);
                            if (w >= 0F && w <= 1F)
                            {
                                pt1 = _RGrid.GetNodePointRelativeToGrid(new GridCell(row, column));
                                pt2 = _RGrid.GetNodePointRelativeToGrid(new GridCell(row + 1, column));
                                rx = pt1.X;
                                ry = (1 - w) * pt1.Y + w * pt2.Y;
                                pt = (ICoordinate)(new Coordinate(rx, ry));
                                _RGrid.TransformRelativeToGlobal(pt);
                            }
                        }
                        break;

                    default:
                        break;
                }

                return pt;
                
            }
            catch (Exception ex)
            {
                throw;
            }
            
        }

        /// <summary>
        /// Marks the branch.
        /// </summary>
        /// <param name="row">The row.</param>
        /// <param name="column">The column.</param>
        /// <param name="branch">The branch.</param>
        /// <remarks></remarks>
        private void MarkBranch(int row, int column, int branch)
        {
            switch (branch)
            {
                case 1:
                    _HBranch[row-1, column-1] = true;
                    break;
                case 2:
                    _VBranch[row-1, column-1] = true;
                    break;
                default:
                    break;
            }
        }

        /// <summary>
        /// Branches the is marked.
        /// </summary>
        /// <param name="row">The row.</param>
        /// <param name="column">The column.</param>
        /// <param name="branch">The branch.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private bool BranchIsMarked(int row, int column, int branch)
        {
            bool result = false;
            switch (branch)
            {
                case 1:
                    result = _HBranch[row-1, column-1];
                    break;
                case 2:
                    result = _VBranch[row-1, column-1];
                    break;
                default:
                    break;
            }
            return result;

        }

        /// <summary>
        /// Determines whether [is no data value] [the specified data value].
        /// </summary>
        /// <param name="dataValue">The data value.</param>
        /// <returns><c>true</c> if [is no data value] [the specified data value]; otherwise, <c>false</c>.</returns>
        /// <remarks></remarks>
        private bool IsNoDataValue(float dataValue)
        {
            // If UseDefaultNoDataRange is set to true, first check to see
            // if dataValue is large enough to be considered a no data cell.
            if (UseDefaultNoDataRange)
            {
                if (dataValue > NoDataRangeMinimum) return true;
            }

            // Next check to see if dataValue matches any of the explict no data values.
            foreach (float val in ExcludedValues)
            { if (val == dataValue) return true; }

            // If it gets this far, dataValue is valid data, so return true.
            return false;
        }

        /// <summary>
        /// Processes the contour level.
        /// </summary>
        /// <param name="level">The level.</param>
        /// <param name="firstRow">The first row.</param>
        /// <param name="firstColumn">The first column.</param>
        /// <param name="lastRow">The last row.</param>
        /// <param name="lastColumn">The last column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private ContourLineList ProcessContourLevel(float level, int firstRow, int firstColumn, int lastRow, int lastColumn)
        {
            ContourLine contourLine;
            ContourLineList contours = new ContourLineList();

            // Initialize branch flags to account for no data values
            InitializeBranchFlags();

            // Get copy of LayerArray and adjust it so that no node value is exactly
            // equal to the contour level.
            float delta = level * 1.00001F;
            if (level == 0F) delta = 1.0E-20F;

            Array2d<float> buffer = new Array2d<float>(LayerArray);
            for (int row = 1; row <= buffer.RowCount; row++)
            {
                for (int column = 1; column <= buffer.ColumnCount; column++)
                { if (buffer[row, column] == level) buffer[row, column] = delta; }
            }

            // Scan for starting points
            for (int row = firstRow; row <= lastRow; row++)
            {
                for (int column = firstColumn; column <= lastColumn; column++)
                {
                    // Check if a point is on the horizontal branch between 
                    // node (row, column) and node (row, column + 1)
                    if (column != lastColumn)
                    {
                        if (ContainsLevel(level, buffer[row, column], buffer[row, column + 1]))
                        {
                            // Check to see if branch is already marked. 
                            // If not, process the point.
                            if (!BranchIsMarked(row, column, 1))
                            {
                                contourLine = ConstructContourLine(buffer, row, column, level, 1, firstRow, firstColumn, lastRow, lastColumn);
                                if (contourLine != null)
                                {
                                    if (contourLine.Contour.Count > 0)
                                        contours.Add(contourLine);
                                }
                            }
                        }
                    }

                    // Check if a point is on the vertical branch between
                    // node (row, column) and node (row + 1, column)
                    if (row != lastRow)
                    {
                        if (ContainsLevel(level, buffer[row, column], buffer[row + 1, column]))
                        {
                            // Check to see if branch is already marked. 
                            // If not, process the point.
                            if (!BranchIsMarked(row, column, 2))
                            {
                                contourLine = ConstructContourLine(buffer, row, column, level, 2, firstRow, firstColumn, lastRow, lastColumn);
                                if (contourLine != null)
                                {
                                    if (contourLine.Contour.Count > 0)
                                        contours.Add(contourLine);
                                }
                            }
                        }
                    }
                }
            }

            return contours;
        }

        /// <summary>
        /// Constructs the contour line.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="initialRow">The initial row.</param>
        /// <param name="initialColumn">The initial column.</param>
        /// <param name="level">The level.</param>
        /// <param name="initialBranch">The initial branch.</param>
        /// <param name="firstRow">The first row.</param>
        /// <param name="firstColumn">The first column.</param>
        /// <param name="lastRow">The last row.</param>
        /// <param name="lastColumn">The last column.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        private ContourLine ConstructContourLine(Array2d<float> buffer, int initialRow, int initialColumn, float level, int initialBranch, int firstRow, int firstColumn, int lastRow, int lastColumn)
        {

            #region Fields
            int row;
            int column;
            int nextRow = 0;
            int nextColumn = 0;
            ICoordinate pt;
            int branch;
            int nextBranch = 0;
            int direction = 0;
            int nextDirection = 0;
            int beginDirection = 0;
            int reverseDirection = 0;
            bool atDeadEnd;
            bool foundPoint;
            bool closeCurve = false;
            ICoordinate startPoint = null;
            List<ICoordinate>[] line = new List<ICoordinate>[2];
            line[0] = new List<ICoordinate>();
            line[1] = new List<ICoordinate>();
            #endregion

            #region Find and initialize starting point

            // Find the coordinates of the starting point
            startPoint = FindCoordPoint(buffer, initialRow, initialColumn, initialBranch, level);
            if (startPoint == null)
                throw new Exception("Error generating contour line starting point.");

            // Mark the branch as occupied.
            MarkBranch(initialRow, initialColumn, initialBranch);

            #endregion

            // Make a maximum of two passes to generate a complete contour.
            // Closed contours will only require one pass. Open contours may
            // require two passes if the starting point is located somewhere withiin 
            // the middle of the grid.
            for (int pass=0; pass<2; pass++)
            {
                row = initialRow;
                column = initialColumn;
                line[pass].Add(new Coordinate(startPoint.X,startPoint.Y));
                branch = initialBranch;
                direction = beginDirection;
                atDeadEnd = false;

                while ( !atDeadEnd )
                {
                    // Reset the foundPoint flag to false to prepare to search 
                    // for the next point.
                    foundPoint = false;

                    #region Search for the next contour point
                    if (branch == 1)
                    {
                        #region Current point is on a horizontal branch
                        // Loop through branches and search for the next contour point
                        for (int i = 0; i < 6; i++)
                        {
                            switch (i)
                            {
                                case 0:
                                    #region Search downward -- right vertical branch
                                    if (direction <= 0 && row != lastRow && column != lastColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row, column + 1], buffer[row + 1, column + 1]))
                                        {
                                            nextDirection = 1;
                                            reverseDirection = 1;
                                            nextBranch = 2;
                                            nextRow = row;
                                            nextColumn = column + 1;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 1:
                                    #region Search downward -- left vertical branch
                                    if (direction <= 0 && row != lastRow)
                                    {
                                        if (ContainsLevel(level, buffer[row, column], buffer[row + 1, column]))
                                        {
                                            nextDirection = -1;
                                            reverseDirection = 1;
                                            nextBranch = 2;
                                            nextRow = row;
                                            nextColumn = column;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 2:
                                    #region Search downward -- horizontal branch below
                                    if (direction <= 0 && row != lastRow && column!= lastColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row + 1, column], buffer[row + 1, column + 1]))
                                        {
                                            nextDirection = -1;
                                            reverseDirection = 1;
                                            nextBranch = 1;
                                            nextRow = row + 1;
                                            nextColumn = column;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 3:
                                    #region Search upward -- left vertical branch
                                    if (direction >= 0 && row != firstRow)
                                    {
                                        if (ContainsLevel(level, buffer[row - 1, column], buffer[row, column]))
                                        {
                                            nextDirection = -1;
                                            reverseDirection = -1;
                                            nextBranch = 2;
                                            nextRow = row - 1;
                                            nextColumn = column;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 4:
                                    #region Search upward -- right vertical branch
                                    if (direction >= 0 && row != firstRow && column != lastColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row - 1, column + 1], buffer[row, column + 1]))
                                        {
                                            nextDirection = 1;
                                            reverseDirection = -1;
                                            nextBranch = 2;
                                            nextRow = row - 1;
                                            nextColumn = column + 1;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 5:
                                    #region Search upward -- horizontal branch above
                                    if (direction >= 0 && row != firstRow && column != lastColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row - 1, column], buffer[row - 1, column + 1]))
                                        {
                                            nextDirection = 1;
                                            reverseDirection = -1;
                                            nextBranch = 1;
                                            nextRow = row - 1;
                                            nextColumn = column;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                default:
                                    break;
                            }

                            // Exit loop if a point was found.
                            if (foundPoint) break;

                        }
                        #endregion
                    }
                    else
                    {
                        #region Current point is on a vertical branch
                        // Loop through branches and search for the next contour point.
                        for (int i = 0; i < 6; i++)
                        {
                            switch (i)
                            {
                                case 0:
                                    #region Search left -- lower horizontal branch
                                    if (direction <= 0 & row < lastRow && column != firstColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row + 1, column - 1], buffer[row + 1, column]))
                                        {
                                            nextDirection = -1;
                                            reverseDirection = 1;
                                            nextBranch = 1;
                                            nextRow = row + 1;
                                            nextColumn = column - 1;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 1:
                                    #region Search left -- upper horizontal branch
                                    if (direction <= 0 & row < lastRow && column != firstColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row, column - 1], buffer[row, column]))
                                        {
                                            nextDirection = 1;
                                            reverseDirection = 1;
                                            nextBranch = 1;
                                            nextRow = row;
                                            nextColumn = column - 1;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 2:
                                    #region Search left -- left vertical branch
                                    if (direction <= 0 & row < lastRow && column != firstColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row, column - 1], buffer[row + 1, column - 1]))
                                        {
                                            nextDirection = -1;
                                            reverseDirection = 1;
                                            nextBranch = 2;
                                            nextRow = row;
                                            nextColumn = column - 1;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 3:
                                    #region Search right -- lower horizontal branch
                                    if (direction >= 0 & row < lastRow && column != lastColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row + 1, column], buffer[row + 1, column + 1]))
                                        {
                                            nextDirection = -1;
                                            reverseDirection = -1;
                                            nextBranch = 1;
                                            nextRow = row + 1;
                                            nextColumn = column;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 4:
                                    #region Search right -- upper horizontal branch
                                    if (direction >= 0 & row < lastRow && column != lastColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row, column], buffer[row, column + 1]))
                                        {
                                            nextDirection = 1;
                                            reverseDirection = -1;
                                            nextBranch = 1;
                                            nextRow = row;
                                            nextColumn = column;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                case 5:
                                    #region Search right -- right vertical branch
                                    if (direction >= 0 & row < lastRow && column != lastColumn)
                                    {
                                        if (ContainsLevel(level, buffer[row, column + 1], buffer[row + 1, column + 1]))
                                        {
                                            nextDirection = 1;
                                            reverseDirection = -1;
                                            nextBranch = 2;
                                            nextRow = row;
                                            nextColumn = column + 1;
                                            if (!BranchIsMarked(nextRow, nextColumn, nextBranch))
                                            { foundPoint = true; }
                                            else
                                            {
                                                if (nextRow == initialRow && nextColumn == initialColumn && nextBranch == initialBranch)
                                                {
                                                    foundPoint = true;
                                                    closeCurve = true;
                                                }
                                            }
                                        }
                                    }
                                    #endregion
                                    break;
                                default:
                                    break;
                            }

                            // Exit loop if a point was found.
                            if (foundPoint) break;
                        }
                        #endregion
                    }
                    #endregion

                    #region Process the results of the search

                    // A point was found, so process it.
                    if (foundPoint)
                    {
                        if (closeCurve)
                        {
                            // A closed curve only will be generated on pass 0.
                            // Close the curve by adding a final point with the same
                            // coordinates as the starting point.
                            line[pass].Add((ICoordinate)(new Coordinate(startPoint.X, startPoint.Y)));
                            
                            // Create a contour line and return it.
                            return new ContourLine(line[pass], level);
                        }
                        else
                        {
                            // Find the coordinates of the point
                            pt = FindCoordPoint(buffer, nextRow, nextColumn, nextBranch, level);
                            if (pt == null)
                                throw new Exception("Error computing contour point.");

                            // If this is the beginning of pass 0, store the reversed direction 
                            // of beginDirection so that it can be used in pass 1 to search in the 
                            // opposite direction if necessary.
                            if (line[pass].Count == 1) beginDirection = reverseDirection;

                            // Add the point to the contour line.
                            line[pass].Add(pt);

                            // Advance the row, column, and branch variables
                            row = nextRow;
                            column = nextColumn;
                            branch = nextBranch;
                            direction = nextDirection;

                            // Mark the branch as used so it will not be used again.
                            MarkBranch(row, column, branch);
                        }
                    }
                    else
                    {
                        // No point was found. Set flag to indicate that a
                        // dead end has been reached.
                        atDeadEnd = true;
                    }

                    #endregion

                }

                // Go back to the original starting point and search in
                // the opposite direction to find any additional points for this 
                // contour line.
            }

            // Combine the contours from line[0] and line[1] into a single contour.
            // First reverse the order of points in line[0]
            line[0].Reverse();

            // Then append the points from line[1] to line[0]. Do not add the duplicate 
            // starting point in line[1]
            if (line[1].Count > 1)
            {
                for (int i = 1; i < line[1].Count; i++)
                {
                    line[0].Add(line[1][i]);
                }
            }

            if (line[0].Count > 1)
            {
                // Create a contour line and return it.
                return new ContourLine(line[0], level);
            }
            else
                return null;

        }

        #endregion

    }
}
