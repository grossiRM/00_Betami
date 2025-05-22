using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.Modpath.IO
{
    public class TimeseriesFileWriter : IDisposable
    {
        private StringBuilder sb;
        private string e15Fmt = "   0.000000E+00";

        public TimeseriesFileWriter()
        {
            sb = new StringBuilder();
        }

        public string CreateTextRecord(TimeseriesRecord record)
        {
            sb.Length = 0;
            sb.Capacity = 200;

            sb.Append(record.TimePoint.ToString().PadLeft(10));
            sb.Append(record.ModflowTimeStep.ToString().PadLeft(7));
            sb.Append(record.Time.ToString(e15Fmt));
            sb.Append(record.ParticleId.ToString().PadLeft(10));
            sb.Append(record.Group.ToString().PadLeft(5));
            sb.Append(record.X.ToString(e15Fmt));
            sb.Append(record.Y.ToString(e15Fmt));
            sb.Append(record.Z.ToString(e15Fmt));
            sb.Append(record.Grid.ToString().PadLeft(5));
            sb.Append(record.Layer.ToString().PadLeft(5));
            sb.Append(record.Row.ToString().PadLeft(5));
            sb.Append(record.Column.ToString().PadLeft(5));
            sb.Append(record.LocalX.ToString(e15Fmt));
            sb.Append(record.LocalY.ToString(e15Fmt));
            sb.Append(record.LocalZ.ToString(e15Fmt));

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
