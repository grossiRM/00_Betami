using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.NTS.Geometries
{
    public interface IVerticalEnvelope
    {
        double MinZ { get; set; }
        double MaxZ { get; set; }
        void ExpandToInclude(double z);
        void ExpandToInclude(IVerticalEnvelope verticalEnvelope);
        void Init(double z);
    }
}
