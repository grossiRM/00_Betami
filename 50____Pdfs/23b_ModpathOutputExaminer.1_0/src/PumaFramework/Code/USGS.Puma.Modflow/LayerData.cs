using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Modflow;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    public class LayerData : IDisposable
    {
        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public LayerData()
        {
            _FileReader = null;
            _HNoFlo = (float)1.0E+30;
            _HDry = (float)1.0E+20;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="hNoFlo"></param>
        /// <param name="hDry"></param>
        public LayerData(string filename, float hNoFlo, float hDry) : this()
        {
            if (!OpenFile(filename, hNoFlo, hDry))
            { throw new ArgumentException("The specified MODFLOW output file could not be opened.\nIt either does not exist or is not a Modflow binary layer file."); }
        }
        #endregion

        #region Public Properties
        private BinaryLayerReader _FileReader = null;
        public BinaryLayerReader FileReader
        {
            get { return _FileReader; }
        }

        private float _HNoFlo;
        public float HNoFlo
        {
            get { return _HNoFlo; }
            set { _HNoFlo = value; }
        }

        private float _HDry;
        public float HDry
        {
            get { return _HDry; }
            set { _HDry = value; }
        }
        #endregion

        #region Public Methods
        public bool OpenFile(string filename, float hNoFlo, float hDry)
        {
            CloseFile();

            BinaryLayerReader reader = new BinaryLayerReader(filename);
            if (reader == null)
            { return false; }

            if (reader.Valid)
            {
                _FileReader = reader;
                HNoFlo = hNoFlo;
                HDry = HDry;
                return true;
            }
            else
            { 
                return false; 
            }

        }
        public void CloseFile()
        {
            if (_FileReader != null)
            {
                _FileReader.Close();
                _FileReader = null;
            }
        }
        public List<float> GetExcludedValues()
        {
            List<float> values = new List<float>();
            values.Add(_HNoFlo);
            if (_HDry != _HNoFlo)
            { values.Add(_HDry); }
            return values;
        }
        #endregion

        #region Private Methods
        private string BuildRecordKey(int stressPeriod, int timeStep, int modelLayer, string dataType)
        {
            string[] parts = new string[4];
            parts[0] = stressPeriod.ToString();
            parts[1] = timeStep.ToString();
            parts[2] = modelLayer.ToString();
            parts[3] = dataType.Trim().ToUpper();
            return string.Join(",", parts);
        }
        #endregion

        #region IDisposable Members

        public void Dispose()
        {
            CloseFile();
        }

        #endregion

    }
}
