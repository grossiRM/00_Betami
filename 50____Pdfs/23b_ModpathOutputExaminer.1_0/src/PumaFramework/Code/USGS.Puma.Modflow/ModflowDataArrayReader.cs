using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.IO;
using USGS.Puma.Utilities;
using USGS.Puma.Modflow;

namespace USGS.Puma.Modflow
{
    public class ModflowDataArrayReader<T>
    {
        #region Private Fields
        private TextArrayIO<T> _TextIO = null;
        #endregion

        #region Constructors
        public ModflowDataArrayReader(StreamReader fileReader, ModflowNameData nameData)
        {
            if (fileReader == null)
                throw new Exception("No data file reader was specified.");

            _TextIO = new TextArrayIO<T>();
            FileReader = fileReader;
            NameData = nameData;

        }
        #endregion

        #region Public Properties
        private StreamReader _FileReader = null;
        public StreamReader FileReader
        {
            get { return _FileReader; }
            set { _FileReader = value; }
        }

        private ModflowNameData _NameData = null;
        public ModflowNameData NameData
        {
            get { return _NameData; }
            set { _NameData = value; }
        }

        #endregion

        #region Public Methods
        public void Read(IModflowDataArray1d<T> arrayData)
        {
            Array1d<T> a = null;

            if (FileReader == null)
                throw new Exception("The data file reader does not exist.");

            if (arrayData == null)
                throw new Exception("The specified arrayData object does not exist.");

            // Set the Modflow NameData
            arrayData.NameData = NameData;

            IArrayControlRecord<T> controlRecord = ReadControlRecord();
            arrayData.SetControlRecordData(controlRecord);

            switch (arrayData.RecordType)
            {
                case ArrayControlRecordType.Constant:
                    break;
                case ArrayControlRecordType.Internal:
                    a = new Array1d<T>(arrayData.ElementCount);
                    if (_TextIO.Read(a, FileReader))
                    {
                        arrayData.DataArray = a;
                    }

                    break;
                case ArrayControlRecordType.External:
                    arrayData.DataArray = null;
                    if (NameData != null)
                    {
                        string fullname = NameData.GetFullFilename(arrayData.FileUnit);
                        a = new Array1d<T>(arrayData.ElementCount);
                        if (_TextIO.Read(a, fullname))
                        {
                            arrayData.DataArray = a;
                        }
                    }

                    break;
                case ArrayControlRecordType.OpenClose:
                    arrayData.DataArray = null;
                    if (NameData != null)
                    {
                        string fullname = NameData.GetFullFilename(arrayData.Filename);
                        a = new Array1d<T>(arrayData.ElementCount);
                        if (_TextIO.Read(a, fullname))
                        {
                            arrayData.DataArray = a;
                        }
                    }
                    break;
                default:
                    break;
            }

        }
        public void Read(IModflowDataArray2d<T> arrayData)
        {
            Array2d<T> a = null;

            if (FileReader == null)
                throw new Exception("The data file reader does not exist.");

            if (arrayData == null)
                throw new Exception("The specified arrayData object does not exist.");

            // Set the Modflow NameData
            arrayData.NameData = NameData;

            IArrayControlRecord<T> controlRecord = ReadControlRecord();
            arrayData.SetControlRecordData(controlRecord);

            switch (arrayData.RecordType)
            {
                case ArrayControlRecordType.Constant:
                    break;
                case ArrayControlRecordType.Internal:
                    a = new Array2d<T>(arrayData.RowCount,arrayData.ColumnCount);
                    if (_TextIO.Read(a, FileReader))
                    {
                        arrayData.DataArray = a;
                    }

                    break;
                case ArrayControlRecordType.External:
                    arrayData.DataArray = null;
                    if (NameData != null)
                    {
                        string fullname = NameData.GetFullFilename(arrayData.FileUnit);
                        a = new Array2d<T>(arrayData.RowCount,arrayData.ColumnCount);
                        if (_TextIO.Read(a, fullname))
                        {
                            arrayData.DataArray = a;
                        }
                    }

                    break;
                case ArrayControlRecordType.OpenClose:
                    arrayData.DataArray = null;
                    if (NameData != null)
                    {
                        string fullname = NameData.GetFullFilename(arrayData.Filename);
                        a = new Array2d<T>(arrayData.RowCount, arrayData.ColumnCount);
                        if (_TextIO.Read(a, fullname))
                        {
                            arrayData.DataArray = a;
                        }
                    }
                    break;
                default:
                    break;
            }


        }

        public IArrayControlRecord<T> ReadControlRecord()
        {
            string line = FileReader.ReadLine();
            IArrayControlRecord<T> controlRecord = new ArrayControlRecord<T>(line);
            if (NameData != null)
            {
                controlRecord.NameData = NameData;
            }
            return controlRecord;
        }

        #endregion
    }
}
