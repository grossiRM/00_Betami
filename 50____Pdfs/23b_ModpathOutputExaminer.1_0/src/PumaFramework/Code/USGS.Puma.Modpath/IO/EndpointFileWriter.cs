using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.Modpath.IO
{
    public class EndpointFileWriter : IDisposable
    {
        private StringBuilder sb;
        private string e15Fmt = "   0.000000E+00";
        
        public EndpointFileWriter()
        {
            sb = new StringBuilder();
        }

        public string CreateTextRecord(EndpointRecord record)
        {
            sb.Length = 0;
            sb.Capacity = 200;

            sb.Append(record.ParticleId.ToString().PadLeft(10));
            sb.Append(record.Group.ToString().PadLeft(5));
            sb.Append(record.Status.ToString().PadLeft(2));
            sb.Append(record.InitialTime.ToString(e15Fmt));
            sb.Append(record.FinalTime.ToString(e15Fmt));
            sb.Append(record.InitialGrid.ToString().PadLeft(5));
            sb.Append(record.InitialLayer.ToString().PadLeft(5));
            sb.Append(record.InitialRow.ToString().PadLeft(5));
            sb.Append(record.InitialColumn.ToString().PadLeft(5));
            sb.Append(record.InitialFace.ToString().PadLeft(2));
            sb.Append(record.InitialZone.ToString().PadLeft(10));
            sb.Append(record.InitialLocalX.ToString(e15Fmt));
            sb.Append(record.InitialLocalY.ToString(e15Fmt));
            sb.Append(record.InitialLocalZ.ToString(e15Fmt));
            sb.Append(record.FinalGrid.ToString().PadLeft(5));
            sb.Append(record.FinalLayer.ToString().PadLeft(5));
            sb.Append(record.FinalRow.ToString().PadLeft(5));
            sb.Append(record.FinalColumn.ToString().PadLeft(5));
            sb.Append(record.FinalFace.ToString().PadLeft(2));
            sb.Append(record.FinalZone.ToString().PadLeft(10));
            sb.Append(record.FinalLocalX.ToString(e15Fmt));
            sb.Append(record.FinalLocalY.ToString(e15Fmt));
            sb.Append(record.FinalLocalZ.ToString(e15Fmt));
            sb.Append(record.FinalX.ToString(e15Fmt));
            sb.Append(record.FinalY.ToString(e15Fmt));
            sb.Append(record.FinalZ.ToString(e15Fmt));
            if (!string.IsNullOrEmpty(record.Label))
            {
                sb.Append(' ');
                sb.Append(record.Label);
            }

            return sb.ToString(0, sb.Length);

        }

        #region IDisposable Members

        public void Dispose()
        {
            // nothing to do yet
        }

        #endregion
    }
}
