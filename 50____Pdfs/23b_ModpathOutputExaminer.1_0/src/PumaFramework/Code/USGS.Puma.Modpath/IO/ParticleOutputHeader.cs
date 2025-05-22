using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public abstract class ParticleOutputHeader
    {
        public abstract int Version
        { get; }

        public abstract int Revision
        { get; }

        protected string _Label;
        public virtual string Label
        {
            get { return _Label; }
            set { _Label = value; }
        }

        protected int _TrackingDirection;
        public virtual int TrackingDirection
        {
            get { return _TrackingDirection; }
            set { _TrackingDirection = value; }
        }

        protected float _ReferenceTime;
        public virtual float ReferenceTime
        {
            get { return _ReferenceTime; }
            set { _ReferenceTime = value; }
        }

        protected List<string> _Comments = new List<string>();
        public virtual List<string> Comments
        {
            get { return _Comments; }
        }

    }
}
