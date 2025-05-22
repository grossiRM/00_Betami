using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public class PathlineHeader : ParticleOutputHeader
    {
        #region Public Methods

        public override int Version
        {
            get { return 6; }
        }

        public override int Revision
        {
            get { return 0; }
        }

        #endregion

    }
}
