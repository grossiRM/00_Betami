using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ModpathOutputExaminer
{
    public class ModpathNameFileItem
    {
        private string _FileType;
        /// <summary>
        /// 
        /// </summary>
        public string FileType
        {
            get { return _FileType; }
            set { _FileType = value; }
        }

        private int _FileUnit;
        /// <summary>
        /// 
        /// </summary>
        public int FileUnit
        {
            get { return _FileUnit; }
            set { _FileUnit = value; }
        }

        private string _Filename;
        /// <summary>
        /// 
        /// </summary>
        public string Filename
        {
            get { return _Filename; }
            set { _Filename = value; }
        }

    }
}
