using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modflow
{
    public class BcfFileData
    {
        
        #region Constructors
        public BcfFileData(int layerCount, int rowCount, int columnCount)
        {
            Initialize(layerCount, rowCount, columnCount);
        }
        #endregion

        #region Public Properties
        private int _LayerCount;
        /// <summary>
        /// 
        /// </summary>
        public int LayerCount
        {
            get { return _LayerCount; }
        }

        private int _RowCount;
        /// <summary>
        /// 
        /// </summary>
        public int RowCount
        {
            get { return _RowCount; }
        }

        private int _ColumnCount;
        /// <summary>
        /// 
        /// </summary>
        public int ColumnCount
        {
            get { return _ColumnCount; }
        }

        private List<string> _Comments = null;
        /// <summary>
        /// 
        /// </summary>
        public List<string> Comments
        {
            get { return _Comments; }
        }

        private int _IBcfCb;
        /// <summary>
        /// 
        /// </summary>
        public int IBcfCb
        {
            get { return _IBcfCb; }
            set { _IBcfCb = value; }
        }

        private float _HDry;
        /// <summary>
        /// 
        /// </summary>
        public float HDry
        {
            get { return _HDry; }
            set { _HDry = value; }
        }

        private int _IWdFlg;
        /// <summary>
        /// 
        /// </summary>
        public int IWdFlg
        {
            get { return _IWdFlg; }
            set { _IWdFlg = value; }
        }

        private float _WetFct;
        /// <summary>
        /// 
        /// </summary>
        public float WetFct
        {
            get { return _WetFct; }
            set { _WetFct = value; }
        }

        private int _IWetIt;
        /// <summary>
        /// 
        /// </summary>
        public int IWetIt
        {
            get { return _IWetIt; }
            set { _IWetIt = value; }
        }

        private int _IHdWet;
        /// <summary>
        /// 
        /// </summary>
        public int IHdWet
        {
            get { return _IHdWet; }
            set { _IHdWet = value; }
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
            _IBcfCb = 0;
        }

        #endregion

    }
}
