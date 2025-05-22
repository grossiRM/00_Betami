using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace USGS.Puma.Modpath.IO
{
    public sealed class EndpointFileReader : ParticleOutputFileReader
    {
        #region Private Fields
        private int _Cursor = 0;
        #endregion

        #region Constructors
        public EndpointFileReader(string Filename)
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

        public bool AtEnd
        {
            get
            {
                if (_Header == null)
                { return true; }
                else
                {
                    if (_Cursor < _Header.ExportedParticleCount)
                    { return false; }
                    else
                    { return true; }
                }
            }
        }

        private EndpointHeader _Header = null;
        public EndpointHeader Header
        {
            get { return _Header; }
        }

        public EndpointRecord ReadRecord()
        {
            if (AtEnd)
            { throw new Exception("Attempt to read beyond the last Endpoint record"); }
            else
            {
                // Read record line, then parse it.
                string line = _Reader.ReadLine();
                EndpointRecord record = ParseRecordLine(line);

                // Increment cursor and return the record
                _Cursor++;
                return record;
            }
        }
        public List<EndpointRecord> Read()
        {
            Exception error = null;
            try
            {
                EndpointRecord rec;
                List<EndpointRecord> list = new List<EndpointRecord>();
                while (!AtEnd)
                {
                    rec = ReadRecord();
                    if (rec == null)
                    {
                        error = new Exception("A null record was returned.");
                        throw error;
                    }
                    list.Add(rec);
                }
                return list;
            }
            catch (Exception)
            {
                throw;
            }
        }

        #endregion

        #region Private and Protected Methods
        private EndpointHeader ReadHeader()
        {
            try
            {
                EndpointHeader header = new EndpointHeader();
                string line = _Reader.ReadLine();
                char[] delim = {' '};
                string[] lines = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);
                
                line = line.Trim().ToUpper();
                if (lines[0].ToUpper() != "MODPATH_ENDPOINT_FILE")
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
                string[] tokens = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);
                header.TrackingDirection = int.Parse(tokens[0]);
                header.ParticleCount = int.Parse(tokens[1]);
                header.ExportedParticleCount = int.Parse(tokens[2]);
                header.MaximumID = int.Parse(tokens[3]);
                header.ReferenceTime = float.Parse(tokens[4]);

                line = _Reader.ReadLine();
                tokens = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);
                header.PendingCount = int.Parse(tokens[0]);
                header.ActiveCount = int.Parse(tokens[1]);
                header.NormalTerminatedCount = int.Parse(tokens[2]);
                header.ZoneTerminatedCount = int.Parse(tokens[3]);
                header.UnreleasedCount = int.Parse(tokens[4]);
                header.StrandedCount = int.Parse(tokens[5]);

                line = _Reader.ReadLine();
                int count = int.Parse(line);
                header.ParticleGroups.Clear();
                for (int i = 0; i < count; i++)
                {
                    line = _Reader.ReadLine();
                    header.ParticleGroups.Add(line.Trim());
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
                    throw new Exception("Error reading the endpoint file header");    
                }

                return header;
            }
            catch (Exception error)
            {
                throw error;
            }
        }
        private EndpointRecord ParseRecordLine(string line)
        {
            try
            {
                // Split the record line into the data item tokens
                char[] delim = {' '};
                string[] tokens = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);

                // Create a new record and populate the data fields
                EndpointRecord rec = new EndpointRecord();
                rec.ParticleId = int.Parse(tokens[0]);
                rec.Group = int.Parse(tokens[1]);
                rec.Status = int.Parse(tokens[2]);
                rec.InitialTime = float.Parse(tokens[3]);
                rec.FinalTime = float.Parse(tokens[4]);
                rec.InitialGrid = int.Parse(tokens[5]);
                rec.InitialLayer = int.Parse(tokens[6]);
                rec.InitialRow = int.Parse(tokens[7]);
                rec.InitialColumn = int.Parse(tokens[8]);
                rec.InitialFace = int.Parse(tokens[9]);
                rec.InitialZone = int.Parse(tokens[10]);
                rec.InitialLocalX = float.Parse(tokens[11]);
                rec.InitialLocalY = float.Parse(tokens[12]);
                rec.InitialLocalZ = float.Parse(tokens[13]);
                rec.InitialX = float.Parse(tokens[14]);
                rec.InitialY = float.Parse(tokens[15]);
                rec.InitialZ = float.Parse(tokens[16]);
                rec.FinalGrid = int.Parse(tokens[17]);
                rec.FinalLayer = int.Parse(tokens[18]);
                rec.FinalRow = int.Parse(tokens[19]);
                rec.FinalColumn = int.Parse(tokens[20]);
                rec.FinalFace = int.Parse(tokens[21]);
                rec.FinalZone = int.Parse(tokens[22]);
                rec.FinalLocalX = float.Parse(tokens[23]);
                rec.FinalLocalY = float.Parse(tokens[24]);
                rec.FinalLocalZ = float.Parse(tokens[25]);
                rec.FinalX = float.Parse(tokens[26]);
                rec.FinalY = float.Parse(tokens[27]);
                rec.FinalZ = float.Parse(tokens[28]);
                if (tokens.Length > 29)
                {
                    rec.Label = tokens[29];
                }

                // Return the record
                return rec;
            }
            catch (Exception)
            {
                throw new Exception("Error processing Endpoint record");
            }
        }
        #endregion

    }
}
