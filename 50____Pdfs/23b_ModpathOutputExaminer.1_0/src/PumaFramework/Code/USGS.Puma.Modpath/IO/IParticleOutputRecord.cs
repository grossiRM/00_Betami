using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    /// <summary>
    /// Identifies a class as a particle output record
    /// </summary>
    public interface IParticleOutputRecord
    {
        string CreateSummary();
    }
}
