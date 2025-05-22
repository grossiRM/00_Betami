using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.NTS.Features;

namespace USGS.Puma.UI.MapViewer
{
    public interface IRenderer : IDisposable
    {
        string MaskField { get; set; }
        List<double> MaskValues { get; }
        bool IncludeMaskValues { get; set; }
        bool UseMask { get; set; }
        bool IsValidMaskField(IAttributesTable attributes);
    }
}
