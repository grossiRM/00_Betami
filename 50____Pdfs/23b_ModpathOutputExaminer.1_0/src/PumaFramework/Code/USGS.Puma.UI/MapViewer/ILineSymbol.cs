using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface ILineSymbol : ISymbol
    {
        System.Drawing.Drawing2D.DashStyle DashStyle { get; set; }
        float Width { get; set; }
    }
}
