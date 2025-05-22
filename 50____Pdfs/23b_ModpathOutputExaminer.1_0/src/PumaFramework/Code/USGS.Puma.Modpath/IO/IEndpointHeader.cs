using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public interface IEndpointHeader
    {
        int Version { get; }
        int Revision { get; }
    }
}
