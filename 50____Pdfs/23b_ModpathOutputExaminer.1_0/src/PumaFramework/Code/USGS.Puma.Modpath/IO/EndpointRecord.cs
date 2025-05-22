using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modpath.IO
{
    public class EndpointRecord : IParticleOutputRecord
    {

        #region Public Properties
        private int _ID;
        public int ParticleId
        {
            get
            {
                return _ID;
            }
            set
            {
                _ID = value;
            }
        }

        private int _Group;
        public int Group
        {
            get
            {
                return _Group;
            }
            set
            {
                _Group = value;
            }
        }

        private int _Status;
        public int Status
        {
            get
            {
                return _Status;
            }
            set
            {
                _Status = value;
            }
        }

        private float _InitialTime;
        public float InitialTime
        {
            get
            {
                return _InitialTime;
            }
            set
            {
                _InitialTime = value;
            }
        }

        private int _InitialGrid;
        public int InitialGrid
        {
            get
            {
                return _InitialGrid;
            }
            set
            {
                _InitialGrid = value;
            }
        }

        private int _InitialRow;
        public int InitialRow
        {
            get
            {
                return _InitialRow;
            }
            set
            {
                _InitialRow = value;
            }
        }

        private int _InitialColumn;
        public int InitialColumn
        {
            get
            {
                return _InitialColumn;
            }
            set
            {
                _InitialColumn = value;
            }
        }

        private int _InitialLayer;
        public int InitialLayer
        {
            get
            {
                return _InitialLayer;
            }
            set
            {
                _InitialLayer = value;
            }
        }

        private int _InitialFace;
        public int InitialFace
        {
            get
            {
                return _InitialFace;
            }
            set
            {
                if (value < 0 || value > 6)
                {
                    throw new ArgumentException("The Face value must be in the range 0 to 6.");
                }
                _InitialFace = value;
            }
        }

        private float _InitialLocalX;
        public float InitialLocalX
        {
          get { return _InitialLocalX; }
          set { _InitialLocalX = value; }
        }

        private float _InitialLocalY;
        public float InitialLocalY
        {
            get { return _InitialLocalY; }
            set { _InitialLocalY = value; }
        }

        private float _InitialLocalZ;
        public float InitialLocalZ
        {
            get { return _InitialLocalZ; }
            set { _InitialLocalZ = value; }
        }

        private float _InitialX;
        public float InitialX
        {
            get { return _InitialX; }
            set { _InitialX = value; }
        }

        private float _InitialY;
        public float InitialY
        {
            get { return _InitialY; }
            set { _InitialY = value; }
        }

        private float _InitialZ;
        public float InitialZ
        {
            get { return _InitialZ; }
            set { _InitialZ = value; }
        }

        private int _InitialZone;
        public int InitialZone
        {
            get { return _InitialZone; }
            set { _InitialZone = value; }
        }

        private float _Time;
        public float FinalTime
        {
            get { return _Time; }
            set { _Time = value; }
        }

        private int _Grid;
        public int FinalGrid
        {
            get { return _Grid; }
            set { _Grid = value; }
        }

        private int _Row;
        public int FinalRow
        {
            get { return _Row; }
            set { _Row = value; }
        }

        private int _Column;
        public int FinalColumn
        {
            get { return _Column; }
            set { _Column = value; }
        }

        private int _Layer;
        public int FinalLayer
        {
            get { return _Layer; }
            set { _Layer = value; }
        }

        private int _Face;
        public int FinalFace
        {
            get { return _Face; }
            set { _Face = value; }
        }

        private float _LocalX;
        public float FinalLocalX
        {
            get { return _LocalX; }
            set { _LocalX = value; }
        }

        private float _LocalY;
        public float FinalLocalY
        {
            get { return _LocalY; }
            set { _LocalY = value; }
        }

        private float _LocalZ;
        public float FinalLocalZ
        {
            get { return _LocalZ; }
            set { _LocalZ = value; }
        }

        private float _X;
        public float FinalX
        {
            get { return _X; }
            set { _X = value; }
        }

        private float _Y;
        public float FinalY
        {
            get { return _Y; }
            set { _Y = value; }
        }

        private float _Z;
        public float FinalZ
        {
            get { return _Z; }
            set { _Z = value; }
        }

        private int _Zone;
        public int FinalZone
        {
            get { return _Zone; }
            set { _Zone = value; }
        }

        private string _Label = "";
        public string Label
        {
            get 
            { 
                return _Label; 
            }
            set 
            {
                if (value.Length > 40)
                {
                    _Label = value.Remove(40);
                }
                else
                {
                    _Label = value;
                }
            }
        }

        #endregion

        #region IParticleOutputRecord Members

        public string CreateSummary()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("ID = ").Append(this.ParticleId).AppendLine();
            sb.Append("Group = ").Append(this.Group).AppendLine();
            sb.Append("Status = ").Append(this.Status);
            switch (this.Status)
            {
                case 0:
                    sb.Append(" (Pending)");
                    break;
                case 1:
                    sb.Append(" (Active)");
                    break;

                case 2:
                    sb.Append(" (Normal terminated)");
                    break;

                case 3:
                    sb.Append(" (Zone terminated)");
                    break;

                case 4:
                    sb.Append(" (Unreleased)");
                    break;

                case 5:
                    sb.Append(" (Stranded)");
                    break;
                default:
                    break;
            }
            sb.AppendLine();
            sb.Append("Initial time = ").Append(this.InitialTime).AppendLine();
            sb.Append("Final time = ").Append(this.FinalTime).AppendLine();
            sb.Append("Travel time = ").Append(this.FinalTime - this.InitialTime).AppendLine();
            sb.AppendLine();
            sb.AppendLine("Initial point:");
            sb.Append("Grid = ").Append(this.InitialGrid).AppendLine();
            sb.Append("Layer, Row, Column: ").Append(this.InitialLayer).Append(" ");
            sb.Append(this.InitialRow).Append(" ");
            sb.Append(this.InitialColumn).AppendLine();
            sb.Append("Face = ").Append(this.InitialFace).AppendLine();
            sb.Append("Zone = ").Append(this.InitialZone).AppendLine();
            sb.Append("Coordinates:  Local         Global").AppendLine();
            sb.Append("           X  ").Append(this.InitialLocalX.ToString("0.000000".PadRight(14)));
            sb.Append(this.InitialX.ToString("0.000000E+00".PadRight(14))).AppendLine();
            sb.Append("           Y  ").Append(this.InitialLocalY.ToString("0.000000".PadRight(14)));
            sb.Append(this.InitialY.ToString("0.000000E+00".PadRight(14))).AppendLine();
            sb.Append("           Z  ").Append(this.InitialLocalZ.ToString("0.000000".PadRight(14)));
            sb.Append(this.InitialZ.ToString("0.000000E+00".PadRight(14))).AppendLine();
            sb.AppendLine();
            sb.AppendLine("Final point:");
            sb.Append("Grid = ").Append(this.FinalGrid).AppendLine();
            sb.Append("Layer, Row, Column: ").Append(this.FinalLayer).Append(" ");
            sb.Append(this.FinalRow).Append(" ");
            sb.Append(this.FinalColumn).AppendLine();
            sb.Append("Face = ").Append(this.FinalFace).AppendLine();
            sb.Append("Zone = ").Append(this.FinalZone).AppendLine();
            sb.Append("Coordinates:  Local         Global").AppendLine();
            sb.Append("           X  ").Append(this.FinalLocalX.ToString("0.000000".PadRight(14)));
            sb.Append(this.FinalX.ToString("0.000000E+00".PadRight(14))).AppendLine();
            sb.Append("           Y  ").Append(this.FinalLocalY.ToString("0.000000".PadRight(14)));
            sb.Append(this.FinalY.ToString("0.000000E+00".PadRight(14))).AppendLine();
            sb.Append("           Z  ").Append(this.FinalLocalZ.ToString("0.000000".PadRight(14)));
            sb.Append(this.FinalZ.ToString("0.000000E+00".PadRight(14)));

            // Return string
            return sb.ToString(0, sb.Length);

        }

        #endregion
    }
}
