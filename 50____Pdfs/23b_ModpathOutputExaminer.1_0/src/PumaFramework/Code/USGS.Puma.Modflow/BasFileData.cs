using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modflow
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks></remarks>
    public class BasFileData
    {
        #region Private Fields
        /// <summary>
        /// 
        /// </summary>
        private IModflowDataArray2d<int>[] _IBound = null;
        /// <summary>
        /// 
        /// </summary>
        private IModflowDataArray2d<float>[] _SHead = null;
        #endregion

        #region Constructors
        /// <summary>
        /// Resets this instance.
        /// </summary>
        /// <remarks></remarks>
        private void Reset()
        {
            _AllowFlowBetweenConstantHeads = false;
            _UseFreeFormat = true;
            _IsCrossSection = false;
            _HNoFlo = 0.0f;
            _Comments = new List<string>();
            _IBound = null;
            _SHead = null;
            _Comments = new List<string>();
            _LayerCount = 0;
            _RowCount = 0;
            _ColumnCount = 0;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="T:System.Object"/> class.
        /// </summary>
        /// <remarks></remarks>
        public BasFileData()
        {
            Reset();
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="BasFileData"/> class.
        /// </summary>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="rowCount">The row count.</param>
        /// <param name="columnCount">The column count.</param>
        /// <remarks></remarks>
        public BasFileData(int layerCount, int rowCount, int columnCount)
        {
            Initialize(layerCount, rowCount, columnCount);
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
        private float _HNoFlo = 0.0f;
        /// <summary>
        /// Gets or sets the H no flo.
        /// </summary>
        /// <value>The H no flo.</value>
        /// <remarks></remarks>
        public float HNoFlo
        {
            get { return _HNoFlo; }
            set { _HNoFlo = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private List<string> _Comments = null;
        /// <summary>
        /// Gets the comments.
        /// </summary>
        /// <remarks></remarks>
        public List<string> Comments
        {
            get { return _Comments; }
        }

        /// <summary>
        /// 
        /// </summary>
        private bool _AllowFlowBetweenConstantHeads = false;
        /// <summary>
        /// Gets or sets a value indicating whether [allow flow between constant heads].
        /// </summary>
        /// <value><c>true</c> if [allow flow between constant heads]; otherwise, <c>false</c>.</value>
        /// <remarks></remarks>
        public bool AllowFlowBetweenConstantHeads
        {
            get { return _AllowFlowBetweenConstantHeads; }
            set { _AllowFlowBetweenConstantHeads = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private bool _UseFreeFormat = true;
        /// <summary>
        /// Gets or sets a value indicating whether [use free format].
        /// </summary>
        /// <value><c>true</c> if [use free format]; otherwise, <c>false</c>.</value>
        /// <remarks></remarks>
        public bool UseFreeFormat
        {
            get { return _UseFreeFormat; }
            set { _UseFreeFormat = value; }
        }

        /// <summary>
        /// 
        /// </summary>
        private bool _IsCrossSection = false;
        /// <summary>
        /// Gets or sets a value indicating whether this instance is cross section.
        /// </summary>
        /// <value><c>true</c> if this instance is cross section; otherwise, <c>false</c>.</value>
        /// <remarks></remarks>
        public bool IsCrossSection
        {
            get { return _IsCrossSection; }
            set { _IsCrossSection = value; }
        }


        #endregion

        #region Public Methods
        /// <summary>
        /// Initializes the specified layer count.
        /// </summary>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="rowCount">The row count.</param>
        /// <param name="columnCount">The column count.</param>
        /// <remarks></remarks>
        public void Initialize(int layerCount, int rowCount, int columnCount)
        {
            if (layerCount < 1 || rowCount < 1 || columnCount < 1)
            {
                throw new ArgumentException("The number of layers, rows, and columns must all be greater than 0.");
            }

            Reset();

            _LayerCount = layerCount;
            _RowCount = rowCount;
            _ColumnCount = columnCount;

            _IBound = new IModflowDataArray2d<int>[layerCount];
            for (int i = 0; i < layerCount; i++)
            { _IBound[i] = new ModflowDataArray2d<int>(rowCount, columnCount, 1); }

            _SHead = new IModflowDataArray2d<float>[layerCount];
            for (int i = 0; i < layerCount; i++)
            { _SHead[i] = new ModflowDataArray2d<float>(rowCount, columnCount); }

        }
        /// <summary>
        /// Gets the I bound.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IModflowDataArray2d<int> GetIBound(int layerIndex)
        {
            return _IBound[layerIndex - 1];
        }
        /// <summary>
        /// Sets the I bound.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <param name="dataArray">The data array.</param>
        /// <remarks></remarks>
        public void SetIBound(int layerIndex, IModflowDataArray2d<int> dataArray)
        {
            if (dataArray == null)
                throw new Exception("The specified dataArray object does not exist.");
            if ((dataArray.RowCount != _IBound[layerIndex - 1].RowCount) || (dataArray.ColumnCount != _IBound[layerIndex - 1].ColumnCount))
                throw new Exception("The specified array dimensions are incorrect.");

            IArrayControlRecord<int> controlRec = dataArray as IArrayControlRecord<int>;
            _IBound[layerIndex - 1].SetControlRecordData(controlRec);
            if (_IBound[layerIndex - 1].RecordType != ArrayControlRecordType.Constant)
            { _IBound[layerIndex - 1].DataArray = dataArray.DataArray; }

        }

        /// <summary>
        /// Gets the S head.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <returns></returns>
        /// <remarks></remarks>
        public IModflowDataArray2d<float> GetSHead(int layerIndex)
        {
            return _SHead[layerIndex - 1];
        }
        /// <summary>
        /// Sets the S head.
        /// </summary>
        /// <param name="layerIndex">Index of the layer.</param>
        /// <param name="dataArray">The data array.</param>
        /// <remarks></remarks>
        public void SetSHead(int layerIndex, IModflowDataArray2d<float> dataArray)
        {
            if (dataArray == null)
                throw new Exception("The specified dataArray object does not exist.");
            if ((dataArray.RowCount != _SHead[layerIndex - 1].RowCount) || (dataArray.ColumnCount != _SHead[layerIndex - 1].ColumnCount))
                throw new Exception("The specified array dimensions are incorrect.");

            IArrayControlRecord<float> controlRec = dataArray as IArrayControlRecord<float>;
            _SHead[layerIndex - 1].SetControlRecordData(controlRec);
            if (_SHead[layerIndex - 1].RecordType != ArrayControlRecordType.Constant)
            { _SHead[layerIndex - 1].DataArray = dataArray.DataArray; }

        }


        #endregion


    }
}
