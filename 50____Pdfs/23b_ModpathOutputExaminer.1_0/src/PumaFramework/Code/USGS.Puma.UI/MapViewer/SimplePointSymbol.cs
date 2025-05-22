using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class SimplePointSymbol : IPointSymbol
    {
        /// <summary>
        /// 
        /// </summary>
        public SimplePointSymbol()
            : this(PointSymbolTypes.Circle, System.Drawing.Color.Black, System.Drawing.Color.Black, true, true, 1.0f) { }
        public SimplePointSymbol(PointSymbolTypes symbolType, System.Drawing.Color color, float size)
            : this(symbolType, color, color, true, true, size) { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="symbolType"></param>
        /// <param name="color"></param>
        /// <param name="outlineColor"></param>
        /// <param name="enableOutline"></param>
        /// <param name="isFilled"></param>
        /// <param name="size"></param>
        public SimplePointSymbol(PointSymbolTypes symbolType, System.Drawing.Color color, System.Drawing.Color outlineColor, bool enableOutline, bool isFilled, float size)
        {
            _SymbolType = symbolType;
            _EnableOutline = enableOutline;
            _IsFilled = isFilled;
            _Color = color;
            _OutlineColor = outlineColor;
            _Size = size;
        }
        #region IPointSymbol Members

        private PointSymbolTypes _SymbolType;
        /// <summary>
        /// 
        /// </summary>
        public PointSymbolTypes SymbolType
        {
            get
            {
                return _SymbolType;
            }
            set
            {
                _SymbolType = value;
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

        private bool _IsFilled;
        /// <summary>
        /// 
        /// </summary>
        public bool IsFilled
        {
            get
            {
                return _IsFilled;
            }
            set
            {
                _IsFilled = value;
            }
        }

        private System.Drawing.Color _OutlineColor;
        /// <summary>
        /// 
        /// </summary>
        public System.Drawing.Color OutlineColor
        {
            get
            {
                return _OutlineColor;
            }
            set
            {
                _OutlineColor = value;
            }
        }

        private float _Size;
        /// <summary>
        /// 
        /// </summary>
        public float Size
        {
            get
            {
                return _Size;
            }
            set
            {
                _Size = value;
            }
        }

        #endregion

        #region ISymbol Members

        private System.Drawing.Color _Color;
        /// <summary>
        /// 
        /// </summary>
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

        #endregion

    }
}
