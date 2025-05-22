using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace USGS.Puma.Modpath.IO
{
    public class SimulationData
    {
        #region Public Methods
        private string _WorkingDirectory = "";
        /// <summary>
        /// 
        /// </summary>
        public string WorkingDirectory
        {
            get { return _WorkingDirectory; }
            set { _WorkingDirectory = value; }
        }

        private string _SimulationName = "";
        /// <summary>
        /// 
        /// </summary>
        public string SimulationName
        {
            get { return _SimulationName; }
            set { _SimulationName = value; }
        }

        private string _SimulationFilePath = "";
        /// <summary>
        /// 
        /// </summary>
        public string SimulationFilePath
        {
            get { return _SimulationFilePath; }
            set { _SimulationFilePath = value; }
        }

        private int _SimulationType;
        /// <summary>
        /// 
        /// </summary>
        public int SimulationType
        {
            get { return _SimulationType; }
            set { _SimulationType = value; }
        }

        private int _TrackingDirection;
        /// <summary>
        /// 
        /// </summary>
        public int TrackingDirection
        {
            get { return _TrackingDirection; }
            set { _TrackingDirection = value; }
        }

        private string _NameFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string NameFile
        {
            get { return _NameFile; }
            set { _NameFile = value; }
        }

        private string _ListFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string ListFile
        {
            get { return _ListFile; }
            set { _ListFile = value; }
        }

        private string _EndpointFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string EndpointFile
        {
            get { return _EndpointFile; }
            set { _EndpointFile = value; }
        }

        private string _PathlineFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string PathlineFile
        {
            get { return _PathlineFile; }
            set { _PathlineFile = value; }
        }

        private string _TimeseriesFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string TimeseriesFile
        {
            get { return _TimeseriesFile; }
            set { _TimeseriesFile = value; }
        }

        private string _StartingLocationsFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string StartingLocationsFile
        {
            get { return _StartingLocationsFile; }
            set { _StartingLocationsFile = value; }
        }

        private string _TraceFile = "";
        /// <summary>
        /// 
        /// </summary>
        public string TraceFile
        {
            get { return _TraceFile; }
            set { _TraceFile = value; }
        }

        private List<string> _Comments = new List<string>();
        /// <summary>
        /// 
        /// </summary>
        public List<string> Comments
        {
            get { return _Comments; }
        }

        private List<string> _FileLines = new List<string>();
        /// <summary>
        /// 
        /// </summary>
        public List<string> FileLines
        {
            get { return _FileLines; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public string GetFullPathname(string key)
        {
            string filename = "";

            if (key == "mpnam")
            {
                filename = NameFile;
            }
            else if (key == "mplist")
            {
                filename = ListFile;
            }
            else if (key == "endpoint")
            {
                filename = EndpointFile;
            }
            else if (key == "pathline")
            {
                filename = PathlineFile;
            }
            else if (key == "timeseries")
            {
                filename = TimeseriesFile;
            }
            else if (key == "startinglocations")
            {
                filename = StartingLocationsFile;
            }
            else if (key == "trace")
            {
                filename = TraceFile;
            }

            return ConvertToFullPathname(filename);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public string ConvertToFullPathname(string filename)
        {
            if ((Path.IsPathRooted(filename)) || (string.IsNullOrEmpty(filename)))
            {
                return filename;
            }
            else
            {
                return Path.Combine(this.WorkingDirectory, filename);
            }
        }

        #endregion
    }
}
