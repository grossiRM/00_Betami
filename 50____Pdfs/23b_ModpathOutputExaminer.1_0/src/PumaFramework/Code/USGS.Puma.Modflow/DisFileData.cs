using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.FiniteDifference;
using USGS.Puma.IO;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class DisFileData
    {
        #region Private Fields
        /// <summary>
        /// 
        /// </summary>
        private List<string> _Comments = null;
        /// <summary>
        /// 
        /// </summary>
        private TimeDiscretization _SpList = null;
        /// <summary>
        /// 
        /// </summary>
        private IModflowDataArray2d<float>[] _BottomList = null;
        /// <summary>
        /// 
        /// </summary>
        private IModflowDataArray2d<float>[] _BottomCbList = null;
        /// <summary>
        /// 
        /// </summary>
        private int[] _CbFlags = null;
        /// <summary>
        /// 
        /// </summary>
        private IModflowDataArray1d<float> _DelR = null;
        /// <summary>
        /// 
        /// </summary>
        private IModflowDataArray1d<float> _DelC = null;
        /// <summary>
        /// 
        /// </summary>
        private IModflowDataArray2d<float> _Top = null;
        #endregion

        #region Constructors
        /// <summary>
        /// Resets this instance.
        /// </summary>
        /// <remarks></remarks>
        private void Reset()
        {
            _Comments = new List<string>();
            if (_SpList == null)
            {
                _SpList = new TimeDiscretization();
            }
            else
            {
                _SpList.Clear();
            }
            _BottomList = null;
            _BottomCbList = null;
            _Top = null;
            _CbFlags = null;
            _DelC = null;
            _DelR = null;
            _LayerCount = 0;
            _RowCount = 0;
            _ColumnCount = 0;
            _TimeUnit = 0;
            _LengthUnit = 0;
            _NameData = null;

        }
        /// <summary>
        /// Initializes a new instance of the <see cref="DisFileData"/> class.
        /// </summary>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="rowCount">The row count.</param>
        /// <param name="columnCount">The column count.</param>
        /// <param name="constantDelR">The constant del R.</param>
        /// <param name="constantDelC">The constant del C.</param>
        /// <remarks></remarks>
        public DisFileData(int layerCount, int rowCount, int columnCount, float constantDelR, float constantDelC)
        {
            Reset();

            _LayerCount = layerCount;
            _RowCount = rowCount;
            _ColumnCount = columnCount;

            _CbFlags = new int[LayerCount];

            _DelR = new ModflowDataArray1d<float>(ColumnCount, constantDelR);
            _DelC = new ModflowDataArray1d<float>(RowCount, constantDelC);

            _Top = new ModflowDataArray2d<float>(RowCount, ColumnCount, 0.0f);

            _BottomList = new IModflowDataArray2d<float>[LayerCount];
            for (int i = 0; i < _BottomList.Length; i++)
            { _BottomList[i] = new ModflowDataArray2d<float>(RowCount, ColumnCount, 0.0f); }

            _BottomCbList = new IModflowDataArray2d<float>[LayerCount];

            _SpList.Clear();

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DisFileData"/> class.
        /// </summary>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="rowCount">The row count.</param>
        /// <param name="columnCount">The column count.</param>
        /// <param name="constantDelR">The constant del R.</param>
        /// <param name="constantDelC">The constant del C.</param>
        /// <param name="stressPeriodData">The stress period data.</param>
        /// <remarks></remarks>
        public DisFileData(int layerCount, int rowCount, int columnCount,float constantDelR,float constantDelC, StressPeriod[] stressPeriodData)
            : this(layerCount,rowCount,columnCount,constantDelR,constantDelC)
        {
            _SpList.Clear();
            if (stressPeriodData != null)
            {
                if (stressPeriodData.Length > 0)
                {
                    for (int i = 0; i < stressPeriodData.Length; i++)
                    {
                        if (stressPeriodData[i] != null) _SpList.AddStressPeriod(stressPeriodData[i]);
                    }
                }
            }

        }
        /// <summary>
        /// Initializes a new instance of the <see cref="DisFileData"/> class.
        /// </summary>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="rowCount">The row count.</param>
        /// <param name="columnCount">The column count.</param>
        /// <param name="constantDelR">The constant del R.</param>
        /// <param name="constantDelC">The constant del C.</param>
        /// <param name="stressPeriodData">The stress period data.</param>
        /// <param name="timeUnit">The time unit.</param>
        /// <param name="lengthUnit">The length unit.</param>
        /// <remarks></remarks>
        public DisFileData(int layerCount, int rowCount, int columnCount, float constantDelR,float constantDelC, StressPeriod[] stressPeriodData, int timeUnit, int lengthUnit)
            : this (layerCount, rowCount, columnCount, constantDelR, constantDelC, stressPeriodData)
        {
            TimeUnit = timeUnit;
            LengthUnit = lengthUnit;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="DisFileData"/> class.
        /// </summary>
        /// <param name="delC">The del C.</param>
        /// <param name="delR">The del R.</param>
        /// <param name="top">The top.</param>
        /// <param name="bottoms">The bottoms.</param>
        /// <param name="stressPeriodDefs">The stress period defs.</param>
        /// <param name="timeUnit">The time unit.</param>
        /// <param name="lengthUnit">The length unit.</param>
        /// <remarks></remarks>
        public DisFileData(float[] delC, float[] delR, IModflowDataArray2d<float> top, IModflowDataArray2d<float>[] bottoms, StressPeriod[] stressPeriodDefs, int timeUnit, int lengthUnit) 
        {
            Reset();
            // Add code
        }
        #endregion

        #region Public Properties

        /// <summary>
        /// 
        /// </summary>
        private int _LayerCount;
        /// <summary>
        /// Gets the layer count.
        /// </summary>
        /// <remarks></remarks>
        public int LayerCount
        {
            get { return _LayerCount; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _RowCount;
        /// <summary>
        /// Gets the row count.
        /// </summary>
        /// <remarks></remarks>
        public int RowCount
        {
            get { return _RowCount; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _ColumnCount;
        /// <summary>
        /// Gets the column count.
        /// </summary>
        /// <remarks></remarks>
        public int ColumnCount
        {
            get { return _ColumnCount; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _TimeUnit;
        /// <summary>
        /// Gets or sets the time unit.
        /// </summary>
        /// <value>The time unit.</value>
        /// <remarks></remarks>
        public int TimeUnit
        {
            get { return _TimeUnit; }
            set { _TimeUnit = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private int _LengthUnit;
        /// <summary>
        /// Gets or sets the length unit.
        /// </summary>
        /// <value>The length unit.</value>
        /// <remarks></remarks>
        public int LengthUnit
        {
            get { return _LengthUnit; }
            set { _LengthUnit = value; }
        }

        /// <summary>
        /// Gets or sets the del R.
        /// </summary>
        /// <value>The del R.</value>
        /// <remarks></remarks>
        public IModflowDataArray1d<float> DelR
        {
            get { return _DelR; }
            set
            {
                if (value != null)
                {
                    if (value.ElementCount != _DelR.ElementCount)
                        throw new Exception("Invalid data specified for DelR.");
                    _DelR = value;
                }
            }
        }

        /// <summary>
        /// Gets or sets the del C.
        /// </summary>
        /// <value>The del C.</value>
        /// <remarks></remarks>
        public IModflowDataArray1d<float> DelC
        {
            get { return _DelC; }
            set
            {
                if (value != null)
                {
                    if (value.ElementCount != _DelC.ElementCount)
                        throw new Exception("Invalid data specified for DelC.");
                    _DelC = value;
                }
            }
        }

        /// <summary>
        /// Gets or sets the top.
        /// </summary>
        /// <value>The top.</value>
        /// <remarks></remarks>
        public IModflowDataArray2d<float> Top
        {
            get { return _Top; }
            set
            {
                if (value != null)
                {
                    if ( (value.RowCount != _Top.RowCount) || (value.ColumnCount != _Top.ColumnCount) )
                        throw new Exception("Invalid data specified for Top array.");
                    _Top = value;
                }
            }
        }

        /// <summary>
        /// Gets the stress periods.
        /// </summary>
        /// <remarks></remarks>
        public TimeDiscretization TimeDiscretization
        { get { return _SpList; } }

        /// <summary>
        /// Gets the comments.
        /// </summary>
        /// <remarks></remarks>
        public List<string> Comments
        { get { return _Comments; } }

        /// <summary>
        /// 
        /// </summary>
        private ModflowNameData _NameData = null;
        /// <summary>
        /// Gets or sets the name data.
        /// </summary>
        /// <value>The name data.</value>
        /// <remarks></remarks>
        public ModflowNameData NameData
        {
            get { return _NameData; }
            set { _NameData = value; }
        }

        #endregion

        #region Public Methods
        /// <summary>
        /// Creates the cell centered grid.
        /// </summary>
        /// <returns></returns>
        /// <remarks></remarks>
        public CellCenteredArealGrid CreateCellCenteredGrid()
        {
            CellCenteredArealGrid grid = null;
            bool isSquare = false;
            double constantDelC = Convert.ToDouble(this.DelC.ConstantValue);
            double constantDelR = Convert.ToDouble(this.DelR.ConstantValue);
            if (this.DelR.RecordType == ArrayControlRecordType.Constant)
            {
                if (this.DelC.RecordType == ArrayControlRecordType.Constant)
                {
                    if (constantDelC == constantDelR) isSquare = true;
                }
            }

            if (isSquare)
            {
                grid = new CellCenteredArealGrid(this.RowCount, this.ColumnCount, constantDelR, 0, 0, 0);
            }
            else
            {
                Array1d<float> dr = this.DelR.GetDataArrayCopy(true);
                Array1d<float> dc = this.DelC.GetDataArrayCopy(true);
                grid = new CellCenteredArealGrid(dc, dr);
            }

            return grid;

        }
        /// <summary>
        /// Gets the layer confining bed flag.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public int GetLayerConfiningBedFlag(int layerIndex)
        {
            return _CbFlags[layerIndex - 1];
        }
        /// <summary>
        /// Sets the layer confining bed flag.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <param name="value">The value.</param>
        /// <remarks></remarks>
        public void SetLayerConfiningBedFlag(int layerIndex, int value)
        {
            int val = value;
            if (val != 0) val = 1;

            if (val == _CbFlags[layerIndex - 1]) return;
            _CbFlags[layerIndex - 1] = val;

            if (val == 0)
            { _BottomCbList[layerIndex - 1] = null; }
            else
            { _BottomCbList[layerIndex - 1] = new ModflowDataArray2d<float>(RowCount, ColumnCount); }
            
        }
        /// <summary>
        /// Gets the bottom.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IModflowDataArray2d<float> GetBottom(int layerIndex)
        {
            return _BottomList[layerIndex - 1];
        }
        /// <summary>
        /// Sets the bottom.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <param name="dataArray">The data array.</param>
        /// <remarks></remarks>
        public void SetBottom(int layerIndex, IModflowDataArray2d<float> dataArray)
        {
            if (dataArray == null)
                throw new Exception("The specified dataArray object does not exist.");
            if ((dataArray.RowCount != _BottomList[layerIndex - 1].RowCount) || (dataArray.ColumnCount != _BottomList[layerIndex - 1].ColumnCount))
                throw new Exception("The specified array dimensions are incorrect.");

            IArrayControlRecord<float> controlRec = dataArray as IArrayControlRecord<float>;
            _BottomList[layerIndex - 1].SetControlRecordData(controlRec);
            if (_BottomList[layerIndex - 1].RecordType != ArrayControlRecordType.Constant)
            { _BottomList[layerIndex - 1].DataArray = dataArray.DataArray; }

        }
        /// <summary>
        /// Gets the confining bed bottom.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IModflowDataArray2d<float> GetConfiningBedBottom(int layerIndex)
        {
            if (GetLayerConfiningBedFlag(layerIndex) == 0) return null;
            return _BottomCbList[layerIndex - 1];
        }
        /// <summary>
        /// Sets the confining bed bottom.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <param name="dataArray">The data array.</param>
        /// <remarks></remarks>
        public void SetConfiningBedBottom(int layerIndex, IModflowDataArray2d<float> dataArray)
        {
            if (GetLayerConfiningBedFlag(layerIndex) == 0)
                throw new Exception("The specified layer does not have an underlying quasi-3d confining bed.");

            IArrayControlRecord<float> controlRec = dataArray as IArrayControlRecord<float>;
            _BottomCbList[layerIndex - 1].SetControlRecordData(controlRec);
            if (_BottomCbList[layerIndex - 1].RecordType != ArrayControlRecordType.Constant)
            { _BottomCbList[layerIndex - 1].DataArray = dataArray.DataArray; }

        }
        #endregion

    }
}
