using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.UI.MapViewer
{
    public class EnvelopeLayer
    {
        public void Add(USGS.Puma.NTS.Features.Feature feature)
        {
            if (feature.Geometry == null)
                throw new ArgumentNullException("Argument geometry is null.");
            if (feature.Attributes == null)
                throw new ArgumentNullException("Argument attributes is null.");
        }
    }
}
