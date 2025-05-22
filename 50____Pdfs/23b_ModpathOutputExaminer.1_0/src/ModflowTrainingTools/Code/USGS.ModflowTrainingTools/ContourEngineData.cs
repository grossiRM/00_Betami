using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using USGS.Puma.Modflow;
using System.Drawing;

namespace USGS.ModflowTrainingTools
{
    public enum ContourIntervalOption
    {
        AutomaticConstantInterval = 0,
        SpecifiedConstantInterval = 1,
        SpecifiedContourLevels = 2
    }

    public class ContourEngineData
    {
        public ContourEngineData()
        {
            ContourIntervalOption = ContourIntervalOption.AutomaticConstantInterval;
            UseDefaultNoDataRange = true;
            NoDataRangeMinimum = 1000000.0f;
            ReferenceContour = 0.0f;
            ConstantContourInterval = 1.0f;
            ContourLevels = new List<float>();
            ExcludedValues = new List<float>();
            ContourSourceFile = null;
            SelectedDataLayer = null;
            ContourLineWidth = 1;
            ContourColor = Color.Black;
        }

        private ContourIntervalOption _ContourIntervalOption;
        public ContourIntervalOption ContourIntervalOption
        {
            get { return _ContourIntervalOption; }
            set { _ContourIntervalOption = value; }
        }

        private bool _UseDefaultNoDataRange;
        public bool UseDefaultNoDataRange
        {
            get { return _UseDefaultNoDataRange; }
            set { _UseDefaultNoDataRange = value; }
        }

        private float _NoDataRangeMinimum;
        public float NoDataRangeMinimum
        {
            get { return _NoDataRangeMinimum; }
            set { _NoDataRangeMinimum = value; }
        }

        private float _ReferenceContour;
        public float ReferenceContour
        {
            get { return _ReferenceContour; }
            set { _ReferenceContour = value; }
        }

        private float _ConstantContourInterval;
        public float ConstantContourInterval
        {
            get { return _ConstantContourInterval; }
            set { _ConstantContourInterval = value; }
        }

        private List<float> _ContourLevels = null;
        public List<float> ContourLevels
        {
            get { return _ContourLevels; }
            set { _ContourLevels = value; }
        }

        private List<float> _ExcludedValues = null;
        /// <summary>
        /// 
        /// </summary>
        public List<float> ExcludedValues
        {
            get { return _ExcludedValues; }
            set { _ExcludedValues = value; }
        }

        private BinaryLayerReader _ContourSourceFile = null;
        /// <summary>
        /// 
        /// </summary>
        public BinaryLayerReader ContourSourceFile
        {
            get { return _ContourSourceFile; }
            set { _ContourSourceFile = value; }
        }

        private LayerDataRecordHeader _SelectedDataLayer = null;
        /// <summary>
        /// 
        /// </summary>
        public LayerDataRecordHeader SelectedDataLayer
        {
            get { return _SelectedDataLayer; }
            set { _SelectedDataLayer = value; }
        }

        private Color _ContourColor;
        /// <summary>
        /// 
        /// </summary>
        public Color ContourColor
        {
            get { return _ContourColor; }
            set { _ContourColor = value; }
        }

        private int _ContourLineWidth;
        /// <summary>
        /// 
        /// </summary>
        public int ContourLineWidth
        {
            get { return _ContourLineWidth; }
            set { _ContourLineWidth = value; }
        }

    }
}
