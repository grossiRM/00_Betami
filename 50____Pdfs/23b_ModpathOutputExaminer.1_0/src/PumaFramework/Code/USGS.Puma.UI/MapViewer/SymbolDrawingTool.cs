using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class SymbolDrawingTool : ISymbolDrawingTool, IDisposable
    {
        public SymbolDrawingTool() 
            : this(null, null, true, null) { }
        public SymbolDrawingTool(System.Drawing.Brush fillBrush, System.Drawing.Pen pen)
            : this(fillBrush, pen, true, null) { }
        public SymbolDrawingTool(System.Drawing.Brush fillBrush, System.Drawing.Pen pen, bool enableOutline) 
            : this(fillBrush, pen, enableOutline, null) { }
        public SymbolDrawingTool(System.Drawing.Brush fillBrush, System.Drawing.Pen pen, bool enableOutline, ISymbol symbol)
        {
            _FillBrush = fillBrush;
            _Pen = pen;
            _EnableOutline = true;
            _Symbol = null;
        }
        #region ISymbolDrawingTool Members

        private System.Drawing.Brush _FillBrush;
        /// <summary>
        /// 
        /// </summary>
        public System.Drawing.Brush FillBrush
        {
            get
            {
                return _FillBrush;
            }
            set
            {
                _FillBrush = value;
            }
        }

        private System.Drawing.Pen _Pen;
        /// <summary>
        /// 
        /// </summary>
        public System.Drawing.Pen Pen
        {
            get
            {
                return _Pen;
            }
            set
            {
                _Pen = value;
            }
        }

        private bool _EnableOutline;
        /// <summary>
        /// 
        /// </summary>
        public bool EnableOutline
        {
            get
            {
                return _EnableOutline;
            }
            set
            {
                _EnableOutline = value;
            }
        }

        private ISymbol _Symbol;
        /// <summary>
        /// 
        /// </summary>
        public ISymbol Symbol
        {
            get
            {
                return _Symbol;
            }
            set
            {
                _Symbol = value as ISymbol;
            }
        }

        #endregion

        #region IDisposable Members

        public void Dispose()
        {
            if (_FillBrush != null)
                _FillBrush.Dispose();
            if (_Pen != null)
                _Pen.Dispose();
        }

        #endregion

    }
}
