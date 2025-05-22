using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.NTS.Geometries
{
    public class VerticalEnvelope : IVerticalEnvelope
    {

        public VerticalEnvelope(double z) : this(z, z) { }

        public VerticalEnvelope() : this(0.0, 0.0) { }

        public VerticalEnvelope(double minZ, double maxZ)
        {
            MinZ = minZ;
            MaxZ = maxZ;
        }

        #region IVerticalEnvelope Members

        private double _MinZ = 0.0;
        public double MinZ
        {
            get
            {
                return _MinZ;
            }
            set
            {
                _MinZ = value;
            }
        }

        private double _MaxZ = 0.0;
        public double MaxZ
        {
            get
            {
                return _MaxZ;
            }
            set
            {
                _MaxZ = value;
            }
        }

        public void ExpandToInclude(IVerticalEnvelope verticalEnvelope)
        {
            ExpandToInclude(verticalEnvelope.MinZ);
            ExpandToInclude(verticalEnvelope.MaxZ);
        }

        public void ExpandToInclude(double z)
        {
            this.MinZ = this.MinZ < z ? this.MinZ : z;
            this.MaxZ = this.MaxZ > z ? this.MaxZ : z;
        }

        public void Init(double z)
        {
            MinZ = z;
            MaxZ = z;
        }

        #endregion
    }
}
