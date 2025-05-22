using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class SingleSymbolRenderer : Renderer, ISingleSymbolRenderer
    {
        #region Private Fields
        #endregion

        #region Constructors
        public SingleSymbolRenderer(SymbolType symbolType)
        {
            _SymbolType = symbolType;
            switch (_SymbolType)
            {
                case SymbolType.PointSymbol:
                    _Symbol = new SimplePointSymbol();
                    break;
                case SymbolType.LineSymbol:
                    _Symbol = new LineSymbol();
                    break;
                case SymbolType.FillSymbol:
                    _Symbol = new SolidFillSymbol();
                    break;
                default:
                    break;
            }
        }
        public SingleSymbolRenderer(ISymbol symbol)
        {
            if (symbol == null)
                throw new ArgumentNullException("Symbol argument cannot be null.");

            if (symbol is ILineSymbol)
            {
                _SymbolType = SymbolType.LineSymbol;
                _Symbol = symbol;
            }
            else if (symbol is IFillSymbol)
            {
                _SymbolType = SymbolType.FillSymbol;
                _Symbol = symbol;
            }
            else if (symbol is IPointSymbol)
            {
                _SymbolType = SymbolType.PointSymbol;
                _Symbol = symbol;
            }
            else
            {
                throw new ArgumentException("Invalid symbol.");
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

        #region ISingleSymbolRenderer Members
        private ISymbol _Symbol;
        public ISymbol Symbol
        {
            get
            {
                return _Symbol;
            }
            set
            {
                if (value == null)
                {
                    _Symbol = null;
                }
                else
                {
                    switch (_SymbolType)
                    {
                        case SymbolType.PointSymbol:
                            throw new ArgumentException("PointSymbol type is not yet supported.");
                            break;
                        case SymbolType.LineSymbol:
                            if (value is ILineSymbol)
                            { 
                                _Symbol = value as ISymbol;
                            }
                            else
                            { throw new ArgumentException("Invalid symbol type."); }
                            break;
                        case SymbolType.FillSymbol:
                            if (value is IFillSymbol)
                            { 
                                _Symbol = value as ISymbol;
                            }
                            else
                            { throw new ArgumentException("Invalid symbol type."); }
                            break;
                        default:
                            break;
                    }
                }

            }
        }
        #endregion

        #region IDisposable Members
        public void Dispose()
        {
            // do nothing.
        }
        #endregion

    }
}
