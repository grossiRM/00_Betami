using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public class TimeseriesRecord : IParticleOutputRecord
    {
        #region Public Methods
        private int _TimePoint;
        public int TimePoint
        {
            get { return _TimePoint; }
            set { _TimePoint = value; }
        }

        private int _ModflowTimeStep;
        public int ModflowTimeStep
        {
            get { return _ModflowTimeStep; }
            set { _ModflowTimeStep = value; }
        }

        private float _Time;
        public float Time
        {
            get { return _Time; }
            set { _Time = value; }
        }

        private int _ID;
        public int ParticleId
        {
            get { return _ID; }
            set { _ID = value; }
        }

        private int _Group;
        public int Group
        {
            get { return _Group; }
            set { _Group = value; }
        }

        private float _X;
        public float X
        {
            get { return _X; }
            set { _X = value; }
        }

        private float _Y;
        public float Y
        {
            get { return _Y; }
            set { _Y = value; }
        }

        private float _Z;
        public float Z
        {
            get { return _Z; }
            set { _Z = value; }
        }

        private int _Layer;
        public int Layer
        {
            get { return _Layer; }
            set { _Layer = value; }
        }

        private int _Row;
        public int Row
        {
            get { return _Row; }
            set { _Row = value; }
        }

        private int _Column;
        public int Column
        {
            get { return _Column; }
            set { _Column = value; }
        }

        private int _Grid;
        public int Grid
        {
            get { return _Grid; }
            set { _Grid = value; }
        }

        private float _LocalX;
        public float LocalX
        {
            get { return _LocalX; }
            set { _LocalX = value; }
        }

        private float _LocalY;
        public float LocalY
        {
            get { return _LocalY; }
            set { _LocalY = value; }
        }

        private float _LocalZ;
        public float LocalZ
        {
            get { return _LocalZ; }
            set { _LocalZ = value; }
        }

        #endregion

        #region IParticleOutputRecord Members

        public string CreateSummary()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("Particle ID = ").Append(this.ParticleId).AppendLine().AppendLine();
            sb.Append("Time point = ").Append(this.TimePoint).AppendLine();
            sb.Append("MODFLOW cumulative time step = ").Append(this.ModflowTimeStep).AppendLine();
            sb.Append("Time = ").Append(this.Time).AppendLine();

            // Return string
            return sb.ToString(0, sb.Length);

        }

        #endregion
    }
}
