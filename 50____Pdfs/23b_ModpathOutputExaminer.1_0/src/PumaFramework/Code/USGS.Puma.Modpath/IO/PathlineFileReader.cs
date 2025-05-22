using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.Modpath.IO
{
    public sealed class PathlineFileReader : ParticleOutputFileReader
    {
        #region Private Fields
        #endregion

        #region Constructors
        public PathlineFileReader(string Filename)
        {
            try
            {
                _Reader = new StreamReader(Filename);
                if (_Reader == null)
                {
                    _Valid = false;
                }
                else
                {
                    _Header = ReadHeader();
                    if (_Header == null)
                    {
                        _Valid = false;
                        _Reader.Close();
                        _Reader = null;
                    }
                    else
                    {
                        _Valid = true;
                    }
                }
            }
            catch (FileNotFoundException)
            {
                throw;
            }
            catch (Exception)
            {
                _Valid = false;
                if (_Reader != null)
                {
                    _Reader.Close();
                    _Reader = null;
                }
            }

        }
        #endregion

        #region Public Methods

        private bool _AtEnd = false;
        public bool AtEnd
        {
            get { return _AtEnd; }
        }

        private int _RecordCount = 0;
        public int RecordCount
        {
            get { return _RecordCount; }
        }

        private PathlineHeader _Header = null;
        public PathlineHeader Header
        {
            get { return _Header; }
            set { _Header = value; }
        }

        public PathlineRecord ReadRecord()
        {
            try
            {
                if (AtEnd)
                {
                    throw new Exception("Attempt to read beyond the last record in a pathline file");
                }
                else
                {
                    // Read a record line and trim it.
                    string line = _Reader.ReadLine();
                    line = line.Trim();

                    // Set AtEnd = true if a blank line was read and then
                    // return null. Do not throw an error, because this could
                    // often occur if there are blank lines at the end of the
                    // file. In that case, we just want to disregard them and
                    // treat the situation as an end of file condition.
                    if (string.IsNullOrEmpty(line))
                    {
                        _AtEnd = true;
                        return null;
                    }
                    PathlineRecord record = ParseRecordLine(line);

                    // Set AtEnd = true if the stream is at the end.
                    if (_Reader.EndOfStream)
                    { _AtEnd = true; }

                    // Increment record count
                    _RecordCount++;

                    // Return record
                    return record;
                }
            }
            catch (Exception error)
            {
                throw error;
            }
        }

        public List<PathlineRecord> Read()
        {
            try
            {
                // Check to make sure we are at the start of the records.
                // If individual records have already been read (i.e. _RecordCount > 0),
                // then throw an error.
                if (_RecordCount != 0)
                {
                    string s = "Reading a list of pathline records is not allowed if one or more";
                    s = s + "individual records have already been read.";
                    throw new Exception(s);
                }

                // Loop through all the records and build the records list.
                // If a null record is return skip over it and stop when the
                // AtEnd flag gets set to true.
                PathlineRecord rec;
                List<PathlineRecord> list = new List<PathlineRecord>();
                while (!AtEnd)
                {
                    rec = ReadRecord();
                    if (rec != null)
                    {
                        list.Add(rec);
                    }
                }

                // Return list
                return list;

            }
            catch (Exception error)
            {
                throw new Exception("Error reading a list of pathline records", error);
            }
        }

        #endregion

        #region Private and Protected Methods
        private PathlineHeader ReadHeader()
        {
            try
            {
                PathlineHeader header = new PathlineHeader();

                string line = _Reader.ReadLine();
                char[] delim = { ' ' };
                string[] lines = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);

                line = line.Trim().ToUpper();
                if (lines[0].ToUpper() != "MODPATH_PATHLINE_FILE")
                {
                    return null;
                }
                else
                {
                    if (int.Parse(lines[1]) != header.Version)
                    { return null; }
                    if (int.Parse(lines[2]) != header.Revision)
                    { return null; }
                }
                header.Label = line;

                line = _Reader.ReadLine();
                List<string> tokens = USGS.Puma.Utilities.StringUtility.ParseAsFortranFreeFormat(line, false);
                header.TrackingDirection = int.Parse(tokens[0]);
                if (tokens.Count > 1)
                {
                    header.ReferenceTime = float.Parse(tokens[1]);
                }

                while (true)
                {
                    line = _Reader.ReadLine();
                    if (line[0] == '#')
                    {
                        header.Comments.Add(line);
                    }
                    else
                    {
                        break;
                    }
                }

                line = line.Trim().ToUpper();
                if (line != "END HEADER")
                {
                    throw new Exception("The END HEADER line was not found in the timeseries file.");
                }


                return header;
            }
            catch (Exception error)
            {
                throw new Exception("Error reading the pathline file header", error);
            }
        }
        private PathlineRecord ParseRecordLine(string line)
        {
            try
            {
                // Add code
                // Split the record line into the data item tokens
                char[] delim = { ' ' };
                string[] tokens = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);

                // Create a new record and populate the data fields
                PathlineRecord rec = new PathlineRecord();
                rec.ParticleId = int.Parse(tokens[0]);
                rec.Group = int.Parse(tokens[1]);
                rec.TimePoint = int.Parse(tokens[2]);
                rec.ModflowTimeStep = int.Parse(tokens[3]);
                rec.Time = float.Parse(tokens[4]);
                rec.X = float.Parse(tokens[5]);
                rec.Y = float.Parse(tokens[6]);
                rec.Z = float.Parse(tokens[7]);
                rec.Layer = int.Parse(tokens[8]);
                rec.Row = int.Parse(tokens[9]);
                rec.Column = int.Parse(tokens[10]);
                rec.Grid = int.Parse(tokens[11]);
                rec.LocalX = float.Parse(tokens[12]);
                rec.LocalY = float.Parse(tokens[13]);
                rec.LocalZ = float.Parse(tokens[14]);
                rec.LineNumber = int.Parse(tokens[15]);
                return rec;
            }
            catch (Exception error)
            {
                throw new Exception("Error reading and parsing pathline record", error);
            }
        }

        #endregion

    }
}
