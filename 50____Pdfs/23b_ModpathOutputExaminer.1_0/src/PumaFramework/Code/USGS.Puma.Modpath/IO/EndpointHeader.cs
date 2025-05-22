using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public class EndpointHeader : ParticleOutputHeader
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

        private int _ParticleCount;
        public int ParticleCount
        {
            get { return _ParticleCount; }
            set { _ParticleCount = value; }
        }

        private int _ExportedParticleCount;
        public int ExportedParticleCount
        {
            get { return _ExportedParticleCount; }
            set { _ExportedParticleCount = value; }
        }

        private int _MaximumID;
        public int MaximumID
        {
            get { return _MaximumID; }
            set { _MaximumID = value; }
        }

        private int _PendingCount;
        public int PendingCount
        {
            get { return _PendingCount; }
            set { _PendingCount = value; }
        }

        private int _ActiveCount;
        public int ActiveCount
        {
            get { return _ActiveCount; }
            set { _ActiveCount = value; }
        }

        private int _NormalTerminatedCount;
        public int NormalTerminatedCount
        {
            get { return _NormalTerminatedCount; }
            set { _NormalTerminatedCount = value; }
        }

        private int _ZoneTerminatedCount;
        public int ZoneTerminatedCount
        {
            get { return _ZoneTerminatedCount; }
            set { _ZoneTerminatedCount = value; }
        }

        private int _UnreleasedCount;
        public int UnreleasedCount
        {
            get { return _UnreleasedCount; }
            set { _UnreleasedCount = value; }
        }

        private int _StrandedCount;
        public int StrandedCount
        {
            get { return _StrandedCount; }
            set { _StrandedCount = value; }
        }

        private List<string> _ParticleGroups = new List<string>();
        public List<string> ParticleGroups
        {
            get { return _ParticleGroups; }
        }

        #endregion
    }
}
