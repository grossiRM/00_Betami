using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class ValueSymbolPair
    {
        public ValueSymbolPair(int value, ISymbol symbol)
        {
            Value = value as object;
            Symbol = symbol;
        }
        public ValueSymbolPair(float value, ISymbol symbol)
        {
            Value = value as object;
            Symbol = symbol;
        }
        public ValueSymbolPair(object value, ISymbol symbol)
        {
            Value = value;
            Symbol = symbol;
        }
        private object Value;
        public object Value1
        {
            get { return Value; }
            set { Value = value; }
        }

        private ISymbol Symbol;
        public ISymbol Symbol1
        {
            get { return Symbol; }
            set { Symbol = value; }
        }
    }
}
