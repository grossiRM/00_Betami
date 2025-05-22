using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;

namespace USGS.Puma.UI.MapViewer
{
    public class ColorRampRenderer : Renderer, IColorRampRenderer
    {
        #region Constructors
        public ColorRampRenderer(SymbolType symbolType)
        {
            _SymbolType = symbolType;
            _ColorRamp = null;
            _UseRenderArray = false;
            _RenderArray = null;
            _ExcludedValues = new List<double>();
        }
        public ColorRampRenderer(SymbolType symbolType, IColorRamp colorRamp)
        {
            _SymbolType = symbolType;
            _ColorRamp = colorRamp;
            _UseRenderArray = false;
            _RenderArray = null;
            _ExcludedValues = new List<double>();
        }
        public ColorRampRenderer(SymbolType symbolType, IColorRamp colorRamp, double[] valuesArray)
        {
            _UseRenderArray = true;
            _RenderArray = valuesArray;
            _SymbolType = symbolType;
            _ColorRamp = colorRamp;
            _ExcludedValues = new List<double>();
        }
        public ColorRampRenderer(SymbolType symbolType, IColorRamp colorRamp, float[] valuesArray)
        {
            _UseRenderArray = true;
            SetRenderArray(valuesArray);
            _SymbolType = symbolType;
            _ColorRamp = colorRamp;
            _ExcludedValues = new List<double>();
        }

        #endregion

        #region IColorRampRenderer Members
        private bool _UseRenderArray;
        public bool UseRenderArray
        {
            get { return _UseRenderArray; }
            set { _UseRenderArray = value; }
        }

        private double[] _RenderArray;
        public double[] RenderArray
        {
            get { return _RenderArray; }
            set { _RenderArray = value; }
        }

        public void SetRenderArray(float[] arrayValues)
        {
            if (arrayValues == null)
            { _RenderArray = null; }
            else
            {
                _RenderArray = new double[arrayValues.Length];
                for (int i = 0; i < arrayValues.Length; i++)
                { _RenderArray[i] = Convert.ToDouble(arrayValues[i]); }
            }
        }
        public void SetRenderArray(double[] arrayValues)
        {
            _RenderArray = arrayValues;
        }
        public void SetRenderArray(Array2d<float> arrayValues)
        {
            if (arrayValues == null)
            { _RenderArray = null; }
            else
            {
                float[] values = arrayValues.GetValues();
                SetRenderArray(values);
            }
        }
        public void SetRenderArray(Array2d<double> arrayValues)
        {
            if (arrayValues == null)
            { _RenderArray = null; }
            else
            {
                _RenderArray = arrayValues.GetValues();
            }
        }

        private string _RenderField;
        public string RenderField
        {
            get
            {
                return _RenderField;
            }
            set
            {
                _RenderField = value;
            }
        }

        private IColorRamp _ColorRamp;
        public IColorRamp ColorRamp
        {
            get
            {
                return _ColorRamp;
            }
            set
            {
                _ColorRamp = value;
            }
        }

        private ISymbol _BaseSymbol;
        public ISymbol BaseSymbol
        {
            get
            {
                return _BaseSymbol;
            }
            set
            {
                if (value == null)
                { _BaseSymbol = value; }
                else
                {
                    switch (_SymbolType)
                    {
                        case SymbolType.PointSymbol:
                            if (value is IPointSymbol)
                                _BaseSymbol = value as ISymbol;
                            break;
                        case SymbolType.LineSymbol:
                            if (value is ILineSymbol)
                                _BaseSymbol = value as ISymbol;
                            break;
                        case SymbolType.FillSymbol:
                            if (value is ISolidFillSymbol)
                            { _BaseSymbol = value as ISymbol; }
                            else
                            { throw new NotImplementedException("Fill symbol must be type ISolidFillSymbol."); }
                            break;
                        default:
                            throw new Exception("Invalid symbol type.");
                            break;
                    }
                }
            }
        }

        private List<double> _ExcludedValues;
        public List<double> ExcludedValues
        {
            get { return _ExcludedValues; }
            set { _ExcludedValues = value; }
        }

        #endregion

        #region IDisposable Members

        public void Dispose()
        {
            // do nothing.
        }

        #endregion

        #region IVectorRenderer Members

        private SymbolType _SymbolType;
        public SymbolType SymbolType
        {
            get { return _SymbolType; }
        }

        private double _MinimumValue;
        public double MinimumValue
        {
            get
            {
                return _MinimumValue;
            }
            set
            {
                _MinimumValue = value;
            }
        }

        private double _MaximumValue;
        public double MaximumValue
        {
            get
            {
                return _MaximumValue;
            }
            set
            {
                _MaximumValue = value;
            }
        }

        public float GetPosition(double value)
        {
            double dv = _MaximumValue - _MinimumValue;
            if (dv == 0.0)
            { return 0.5f; }
            else
            { return (float)((value - _MinimumValue) / dv); }
        }

        #endregion


    }
}
