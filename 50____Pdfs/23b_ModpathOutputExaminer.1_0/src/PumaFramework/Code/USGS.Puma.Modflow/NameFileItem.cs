using System;
using System.Collections.Generic;
using System.Text;

namespace USGS.Puma.Modflow
{
    public class NameFileItem : INameFileItem
    {
        #region Constructors
        public NameFileItem()
        {
            FileType = "";
            FileUnit = 0;
            FileName = "";
            FileStatus = "";
        }
        public NameFileItem(string fileType,int fileUnit,string fileName,string fileStatus)
        {
            FileType = fileType;
            FileUnit = fileUnit;
            FileName = fileName;
            FileStatus = fileStatus;
        }

        public NameFileItem(string fileType, int fileUnit, string fileName)
            : this()
        {
            FileType = fileType;
            FileUnit = fileUnit;
            FileName = fileName;
        }

        #endregion

        #region Public Properties
        string _FileType;
        public string FileType
        {
            get { return _FileType; }
            set { _FileType = value; }
        }

        int _FileUnit;
        public int FileUnit
        {
            get { return _FileUnit; }
            set { _FileUnit = value; }
        }

        string _FileName;
        public string FileName
        {
            get { return _FileName; }
            set { _FileName = value; }
        }

        string _FileStatus;
        public string FileStatus
        {
            get { return _FileStatus; }
            set { _FileStatus = value; }
        }
        #endregion


    }
}
