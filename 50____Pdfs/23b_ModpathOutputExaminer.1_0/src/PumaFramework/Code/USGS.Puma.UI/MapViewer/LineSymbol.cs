using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class LineSymbol : ILineSymbol
    {
        #region Static Methods
        public static ISymbolDrawingTool CreateDrawingTool(ILineSymbol symbol)
        {
            System.Drawing.Pen pen = new System.Drawing.Pen(symbol.Color, symbol.Width);
            pen.DashStyle = symbol.DashStyle;
            return new SymbolDrawingTool(null, pen);

        }
        #endregion

        #region Constructors
        public LineSymbol()
        {
            _Color = System.Drawing.Color.Black;
            _Width = 1.0f;
            _DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }
        public LineSymbol(System.Drawing.Color color, System.Drawing.Drawing2D.DashStyle dashStyle, float width)
        {
            _Color = color;
            _Width = width;
            _DashStyle = dashStyle;
        }
        #endregion

        #region ILineSymbol Members

        private System.Drawing.Color _Color;
        public System.Drawing.Color Color
        {
            get
            {
                return _Color;
            }
            set
            {
                _Color = value;
            }
        }

        private System.Drawing.Drawing2D.DashStyle _DashStyle;
        public System.Drawing.Drawing2D.DashStyle DashStyle
        {
            get
            {
                return _DashStyle;
            }
            set
            {
                _DashStyle = value;
            }
        }

        private float _Width;
        public float Width
        {
            get
            {
                return _Width;
            }
            set
            {
                _Width = value;
            }
        }

        #endregion



    }
}
