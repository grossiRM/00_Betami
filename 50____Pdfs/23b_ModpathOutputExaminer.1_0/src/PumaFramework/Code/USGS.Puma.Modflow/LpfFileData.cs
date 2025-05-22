using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modflow
{
    public class LpfFileData
    {
        #region Private Fields
        
        
        #endregion

        #region Constructors
        public LpfFileData(int layerCount, int rowCount, int columnCount)
        {
            Initialize(layerCount, rowCount, columnCount);
        }
        #endregion

        #region Public Properties
        private int _LayerCount;
        public int LayerCount
        {
            get { return _LayerCount; }
        }

        private int _RowCount;
        public int RowCount
        {
            get { return _RowCount; }
        }

        private int _ColumnCount;
        public int ColumnCount
        {
            get { return _ColumnCount; }
        }

        /// <summary>
        /// 
        /// </summary>
        private List<string> _Comments = null;
        public List<string> Comments
        {
            get { return _Comments; }
        }

        private int _ILpfCb;
        public int ILpfCb
        {
            get { return _ILpfCb; }
            set { _ILpfCb = value; }
        }

        private float _HDry;
        public float HDry
        {
            get { return _HDry; }
            set { _HDry = value; }
        }

        private int _NpLpf;
        public int NpLpf
        {
            get { return _NpLpf; }
            set { _NpLpf = value; }
        }

        private bool _UseStorageCoefficient;
        public bool UseStorageCoefficient
        {
            get { return _UseStorageCoefficient; }
            set { _UseStorageCoefficient = value; }
        }

        private bool _UseConstantCV;
        public bool UseConstantCV
        {
            get { return _UseConstantCV; }
            set { _UseConstantCV = value; }
        }

        private bool _UseStartingHeadToComputeThickness;
        public bool UseStartingHeadToComputeThickness
        {
            get { return _UseStartingHeadToComputeThickness; }
            set { _UseStartingHeadToComputeThickness = value; }
        }

        private bool _NoVerticalCvCorrection;
        public bool NoVerticalCvCorrection
        {
            get { return _NoVerticalCvCorrection; }
            set { _NoVerticalCvCorrection = value; }
        }

        #endregion

        #region Public Methods
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
            _Comments = new List<string>();

        }

        #endregion

        #region Private Methods
        private void Reset()
        {
            _Comments = new List<string>();
            _LayerCount = 0;
            _RowCount = 0;
            _ColumnCount = 0;
            _HDry = 0.0f;
            _ILpfCb = 0;
            _NpLpf = 0;
            _NoVerticalCvCorrection = false;
            _UseConstantCV = false;
            _UseStartingHeadToComputeThickness = false;
            _UseStorageCoefficient = false;
        }

        #endregion
    }
}
