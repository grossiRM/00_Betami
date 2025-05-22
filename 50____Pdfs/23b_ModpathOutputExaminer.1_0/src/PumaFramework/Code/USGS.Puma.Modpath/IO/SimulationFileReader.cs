using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace USGS.Puma.Modpath.IO
{
    public class SimulationFileReader
    {
        #region Constructors
        public SimulationFileReader(string filename)
        {
            if (File.Exists(filename))
            {
                if (Path.IsPathRooted(filename))
                {
                    WorkingDirectory = Path.GetDirectoryName(filename);
                    Filename = filename;
                }
                else
                {
                    throw new ArgumentException("The specified filename is not a fully rooted pathname.");
                }
            }
            else
            {
                string message = "The MODPATH simulation file does not exist: ";
                message = message + "\n" + filename;
                throw new FileNotFoundException(message);
            }
        }
        #endregion

        #region Private Fields

        #endregion

        #region Public Methods
        private string _WorkingDirectory;
        public string WorkingDirectory
        {
            get { return _WorkingDirectory; }
            protected set { _WorkingDirectory = value; }
        }

        private string _Filename;
        public string Filename
        {
            get { return _Filename; }
            protected set { _Filename = value; }
        }

        public SimulationData Read()
        {
            SimulationData data = new SimulationData();
            using (StreamReader reader = new StreamReader(Filename))
            {
                // Set properties of the simulation and simulation file
                data.WorkingDirectory = WorkingDirectory;
                data.SimulationFilePath = Filename;
                data.SimulationName = Path.GetFileNameWithoutExtension(Filename);
                // Read comments
                string line = ReadComments(data.Comments, reader);

                // Read name file and listing file names
                data.NameFile = line.Trim();
                data.ListFile = reader.ReadLine().Trim();
                
                // Read options line
                line = reader.ReadLine();
                char[] delim = { ' ' };
                string[] tokens = line.Split(delim, StringSplitOptions.RemoveEmptyEntries);
                data.SimulationType = int.Parse(tokens[0]);
                data.TrackingDirection = int.Parse(tokens[1]);

                // Read endpoint file name
                data.EndpointFile = reader.ReadLine().Trim();

                // Read pathline or timeseries file name
                if (data.SimulationType == 2)
                { data.PathlineFile = reader.ReadLine().Trim(); }
                else if (data.SimulationType == 3)
                { data.TimeseriesFile = reader.ReadLine().Trim(); }
            }
            return data;
        }

        #endregion

        #region Private and Protected Methods
        private string ReadComments(List<string> list, StreamReader reader)
        {
            string line;
            while (true)
            {
                line = reader.ReadLine();
                if (line[0] == '#')
                {
                    list.Add(line);
                }
                else
                { return line; }
            }
        }

        #endregion

    }
}
