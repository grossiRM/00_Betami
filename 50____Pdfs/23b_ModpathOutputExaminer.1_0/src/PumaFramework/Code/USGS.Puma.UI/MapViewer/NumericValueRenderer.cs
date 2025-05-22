using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class NumericValueRenderer : Renderer, INumericValueRenderer
    {

        #region Static Methods
        public static System.Drawing.Color[] GenerateRandomColors(int valueCount, int seed)
        {
            if (valueCount < 1)
            { return null; }

            Random random = new Random(seed);

            System.Drawing.Color[] colors = new System.Drawing.Color[valueCount];
            for (int i = 0; i < colors.Length; i++)
            {
                colors[i] = GetRandomColorStatic(random);
            }

            return colors;

        }
        private static System.Drawing.Color GetRandomColorStatic(Random random)
        {
            int r = GenerateRandomColorComponentStatic(random);
            int g = GenerateRandomColorComponentStatic(random);
            int b = GenerateRandomColorComponentStatic(random);
            return System.Drawing.Color.FromArgb(r, g, b);
        }
        private static int GenerateRandomColorComponentStatic(Random random)
        {
            return Convert.ToInt32(Math.Round(255.0 * random.NextDouble(), 0));
        }
        #endregion

        #region Private Fields
        private Random _Random = null;
        private List<double> _Values = null;
        private List<ISymbol> _Symbols = null;
        #endregion

        #region Constructors
        public NumericValueRenderer(SymbolType symbolType)
        {
            _ExcludedValues = new List<double>();
            _SymbolType = symbolType;
            _Values = new List<double>();
            _Symbols = new List<ISymbol>();
        }
        public NumericValueRenderer(SymbolType symbolType, double[] values) 
            : this(symbolType)
        {
            if (values != null)
            {
                for (int i = 0; i < values.Length; i++)
                {
                    AddValue(values[i]);
                }
            }
        }
        public NumericValueRenderer(SymbolType symbolType, double[] values, ISymbol[] symbols)
            : this(symbolType)
        {
            if (values != null && symbols != null)
            {
                if (values.Length == symbols.Length)
                {
                    for (int i = 0; i < values.Length; i++)
                    {
                        AddValue(values[i]);
                    }
                }
                else
                { throw new ArgumentException("The values and symbols arrays are not the same size."); }
            }
            else
            { throw new ArgumentException("The values and symbols arrays cannot be null."); }
        }
        public NumericValueRenderer(SymbolType symbolType, double[] values, ISymbol[] symbols, int seed) 
            : this(symbolType)
        {
            if (values != null && symbols != null)
            {
                if (values.Length == symbols.Length)
                {
                    System.Drawing.Color[] colors = GenerateRandomColorArray(values.Length, seed);
                    for (int i = 0; i < values.Length; i++)
                    {
                        symbols[i].Color = colors[i];
                        AddValue(values[i], symbols[i]);
                    }
                }
                else
                { throw new ArgumentException("The values and symbols arrays are not the same size."); }
            }
            else
            { throw new ArgumentException("The values and symbols arrays cannot be null."); }

        }
        #endregion

        #region INumericValueRenderer Members

        private string _RenderField = "";
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

        public int ValueCount
        {
            get { return _Values.Count; }
        }

        public void AddValue(int value)
        {
            AddValue((double)value);
        }
        public void AddValue(int value, ISymbol symbol)
        {
            AddValue((double)value, symbol);
        }
        public void AddValue(float value)
        {
            AddValue((double)value);
        }
        public void AddValue(float value, ISymbol symbol)
        {
            AddValue((double)value, symbol);
        }
        public void AddValue(double value)
        {
            ISymbol symbol = null;
            switch (_SymbolType)
            {
                case SymbolType.PointSymbol:
                    symbol = (new SimplePointSymbol(PointSymbolTypes.Circle, GetRandomColor(), 1.0f)) as ISymbol;
                    break;
                case SymbolType.LineSymbol:
                    symbol = (new LineSymbol(GetRandomColor(), System.Drawing.Drawing2D.DashStyle.Solid, 1.0f)) as ISymbol;
                    break;
                case SymbolType.FillSymbol:
                    symbol = (new SolidFillSymbol(GetRandomColor())) as ISymbol;
                    break;
                default:
                    throw new NotImplementedException();
                    break;
            }
            AddValue(value, symbol);
        }
        public void AddValue(double value, ISymbol symbol)
        {
            if (_Values.IndexOf(value) < 0)
            {
                bool valid = false;
                switch (_SymbolType)
                {
                    case SymbolType.PointSymbol:
                        valid = (symbol is IPointSymbol);
                        break;
                    case SymbolType.LineSymbol:
                        valid = (symbol is ILineSymbol);
                        break;
                    case SymbolType.FillSymbol:
                        valid = (symbol is IFillSymbol);
                        break;
                    default:
                        throw new NotImplementedException();
                        break;
                }
                if (!valid)
                    throw new ArgumentException("Invalid symbol type.");

                _Values.Add(value);
                _Symbols.Add(symbol);
            }
            else
            { throw new ArgumentException("The specified value already exists."); }
        }

        public int IndexOf(int value)
        {
            return _Values.IndexOf((double)value);
        }
        public int IndexOf(float value)
        {
            return _Values.IndexOf((double)value);
        }
        public int IndexOf(double value)
        {
            return _Values.IndexOf(value);
        }

        public ISymbol GetSymbol(int index)
        {
            return _Symbols[index];
        }

        public double GetValue(int index)
        {
            return _Values[index];
        }

        public double[] Values
        {
            get { return _Values.ToArray(); }
        }

        public ISymbol[] Symbols
        {
            get { return _Symbols.ToArray(); }
        }

        private List<double> _ExcludedValues;
        public List<double> ExcludedValues
        {
            get { return _ExcludedValues; }
        }

        public void Clear()
        {
            _Values.Clear();
            _Symbols.Clear();
        }

        public void RemoveAt(int index)
        {
            if (index < 0)
                return;
            if (index > _Values.Count - 1)
                return;

            if (_Values.Count == _Symbols.Count)
            {
                _Values.RemoveAt(index);
                _Symbols.RemoveAt(index);
            }
        }

        #endregion

        #region IVectorRenderer Members
        private SymbolType _SymbolType;
        public SymbolType SymbolType
        {
            get { return _SymbolType; }
        }
        #endregion

        #region IDisposable Members

        public void Dispose()
        {
            // do nothing.
        }

        #endregion

        #region Private Methods
        private System.Drawing.Color GetRandomColor()
        {
            int r = GenerateRandomColorComponent();
            int g = GenerateRandomColorComponent();
            int b = GenerateRandomColorComponent();
            return System.Drawing.Color.FromArgb(r, g, b);
        }
        private int GenerateRandomColorComponent()
        {
            if (_Random == null)
                _Random = new Random();
            return Convert.ToInt32(Math.Round(255.0 * _Random.NextDouble(), 0));
        }
        private System.Drawing.Color[] GenerateRandomColorArray(int valueCount, int seed)
        {
            if (valueCount < 1)
            { return null; }

            _Random = new Random(seed);

            System.Drawing.Color[] colors = new System.Drawing.Color[valueCount];
            for (int i = 0; i < colors.Length; i++)
            {
                colors[i] = GetRandomColor();
            }

            return colors;

        }
        private System.Drawing.Color[] GenerateRandomColorArray(int valueCount)
        {
            if (valueCount < 1)
            { return null; }

            System.Drawing.Color[] colors = new System.Drawing.Color[valueCount];
            for (int i = 0; i < colors.Length; i++)
            {
                colors[i] = GetRandomColor();
            }

            return colors;

        }

        #endregion


        #region INumericValueRenderer Members



        #endregion
    }
}
