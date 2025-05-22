using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface IHatchFillSymbol : IFillSymbol
    {
        System.Drawing.Drawing2D.HatchStyle HatchStyle { get; set; }
        System.Drawing.Color BackgroundColor { get; set; }
    }
}
