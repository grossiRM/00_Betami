using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public interface IColorRamp
    {
        System.Drawing.Color[] Colors { get; set; }
        System.Drawing.Color GetColor(float position);
    }
}
