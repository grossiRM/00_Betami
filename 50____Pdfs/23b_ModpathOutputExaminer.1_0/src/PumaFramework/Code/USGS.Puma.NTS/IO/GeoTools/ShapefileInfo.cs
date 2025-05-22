using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace USGS.Puma.NTS.IO
{
    public class ShapefileInfo
    {
        #region Static Methods
        public static bool ShapefileExists(string directory, string basename)
        {
            string s = System.IO.Path.Combine(directory, basename);

            string filename = s + ".shp";
            if (!System.IO.File.Exists(filename))
            { return false; }

            filename = s + ".dbf";
            if (!System.IO.File.Exists(filename))
            { return false; }

            filename = s + ".shx";
            if (!System.IO.File.Exists(filename))
            { return false; }

            return true;

        }

        public static bool TryDelete(string directory, string basename)
        {
            string name = System.IO.Path.Combine(directory, basename);
            string shpName = name + ".shp";

            // If a file with the extension .shp exists, then delete it and
            // all of the other associated files (.dbf, .shx, .sbn, .prj, .shp.xml)
            // If the .shp file does not exist, do not delete anything.
            if (System.IO.File.Exists(shpName))
            {
                System.IO.FileInfo finfo;

                string filename = System.IO.Path.Combine(name, ".dbf");
                if (System.IO.File.Exists(filename))
                {
                    try
                    {
                        finfo = new System.IO.FileInfo(filename);
                        finfo.Delete();
                    }
                    catch
                    {
                        return false;
                    }
                }

                filename = System.IO.Path.Combine(name, ".sbn");
                if (System.IO.File.Exists(filename))
                {
                    try
                    {
                        finfo = new System.IO.FileInfo(filename);
                        finfo.Delete();
                    }
                    catch
                    {
                        return false;
                    }
                }

                filename = System.IO.Path.Combine(name, ".prj");
                if (System.IO.File.Exists(filename))
                {
                    try
                    {
                        finfo = new System.IO.FileInfo(filename);
                        finfo.Delete();
                    }
                    catch
                    {
                        return false;
                    }
                }

                filename = System.IO.Path.Combine(name, ".shx");
                if (System.IO.File.Exists(filename))
                {
                    try
                    {
                        finfo = new System.IO.FileInfo(filename);
                        finfo.Delete();
                    }
                    catch
                    {
                        return false;
                    }
                }

                if (System.IO.File.Exists(shpName))
                {
                    try
                    {
                        finfo = new System.IO.FileInfo(shpName);
                        finfo.Delete();
                    }
                    catch
                    {
                        return false;
                    }
                }

                filename = System.IO.Path.Combine(shpName, ".xml");
                if (System.IO.File.Exists(filename))
                {
                    try
                    {
                        finfo = new System.IO.FileInfo(filename);
                        finfo.Delete();
                    }
                    catch
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return true;
            }

        }

        #endregion

        #region Constructors
        public ShapefileInfo(string directory, string basename)
        {
            Directory = directory;
            Basename = Basename;
            HasBinaryIndex = false;
            HasMetadata = false;
            HasProjection = false;
            Exists = ShapefileInfo.ShapefileExists(Directory, Basename);
            if (Exists)
            {
                string s = System.IO.Path.Combine(Directory, Basename);
                string filename = s + ".sbn";
                HasBinaryIndex = System.IO.File.Exists(filename);

                filename = s + ".prj";
                HasProjection = System.IO.File.Exists(filename);

                filename = s + ".shp.xml";
                HasMetadata = System.IO.File.Exists(filename);
            }
        }
        #endregion

        #region Public Methods
        private string _Directory;
        public string Directory
        {
            get { return _Directory; }
            private set { _Directory = value; }
        }

        private string _Basename;
        public string Basename
        {
            get { return _Basename; }
            private set { _Basename = value; }
        }

        private bool _Exists;
        public bool Exists
        {
            get { return _Exists; }
            private set { _Exists = value; }
        }

        private bool _HasBinaryIndex;
        public bool HasBinaryIndex
        {
            get { return _HasBinaryIndex; }
            private set { _HasBinaryIndex = value; }
       }

        private bool _HasMetadata;
        public bool HasMetadata
        {
            get { return _HasMetadata; }
            private set { _HasMetadata = value; }
        }

        private bool _HasProjection;
        public bool HasProjection
        {
            get { return _HasProjection; }
            private set { _HasProjection = value; }
        }

        public void Delete()
        {
            bool result = ShapefileInfo.TryDelete(Directory, Basename);
        }

        #endregion
    }
}
