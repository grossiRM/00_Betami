using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.IO;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    public class ModflowDataArray2d<T> : IModflowDataArray<T>, IModflowDataArray2d<T>
    {
        #region Private Fields
        private IArrayControlRecord<T> _ControlRecord = null;
        #endregion

        #region Constructors
        public ModflowDataArray2d(int rowCount, int columnCount)
        {
            if ( (rowCount < 1) || (columnCount < 1) )
                throw new Exception("The row and column count both must be greater than 0.");

            _ControlRecord = new ArrayControlRecord<T>();
            _ControlRecord.RecordType = ArrayControlRecordType.Constant;
            _RowCount = rowCount;
            _ColumnCount = columnCount;

        }
        public ModflowDataArray2d(int rowCount, int columnCount, T constantValue)
        {
            if ( (rowCount < 1) || (columnCount < 1) )
                throw new Exception("The element count must be greater than 0.");

            _ControlRecord = new ArrayControlRecord<T>();
            _ControlRecord.SetAsConstant(constantValue);
            _RowCount = rowCount;
            _ColumnCount = columnCount;

        }
        public ModflowDataArray2d(Array2d<T> dataArray)
        {
            if (dataArray == null)
                throw new Exception("The specified dataArray has a null value.");

            _ControlRecord = new ArrayControlRecord<T>();
            _ControlRecord.RecordType = ArrayControlRecordType.Internal;
            _RowCount = dataArray.RowCount;
            _ColumnCount = dataArray.ColumnCount;
            _DataArray = dataArray;
        }
        #endregion

        #region IModflowDataArray<T> Members
        /// <summary>
        /// 
        /// </summary>
        public int ArrayDimension
        {
            get { return 2; }
        }



        #endregion

        #region IModflowDataArray2d<T> Members

        private int _RowCount = 0;
        public int RowCount
        {
            get { return _RowCount; }
        }

        private int _ColumnCount = 0;
        public int ColumnCount
        {
            get { return _ColumnCount; }
        }

        private Array2d<T> _DataArray = null;
        /// <summary>
        /// 
        /// </summary>
        public Array2d<T> DataArray
        {
            get
            {
                return _DataArray;
            }
            set
            {
                if (value == null)
                {
                    // If control record type is CONSTANT and the data array is going 
                    // to be set to variable array, then reset the control record type
                    // to INTERNAL and use default values for associated parameters.
                    if (_ControlRecord.RecordType == ArrayControlRecordType.Constant)
                    { _ControlRecord.RecordType = ArrayControlRecordType.Internal; }

                    // Set the data array.
                    _DataArray = null;
                }
                else
                {
                    if ((value.RowCount == RowCount) && (value.ColumnCount == ColumnCount))
                    { 
                        _DataArray = value;
                    }
                    else
                    { throw new Exception("The size of the specified array is incorrect."); }
                }

            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Array2d<T> GetDataArrayCopy(bool applyMultiplier)
        {
            return CreateProcessedArray(applyMultiplier, null, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public Array2d<T> GetDataArrayCopy(System.IO.StreamReader reader, bool applyMultiplier)
        {
            return CreateProcessedArray(applyMultiplier, reader, null);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public Array2d<T> GetDataArrayCopy(System.IO.BinaryReader reader, bool applyMultiplier)
        {
            return CreateProcessedArray(applyMultiplier, null, reader);
        }

        #endregion

        #region IArrayControlRecord<T> Members
        /// <summary>
        /// 
        /// </summary>
        public ModflowNameData NameData
        {
            get
            {
                return _ControlRecord.NameData;
            }
            set
            {
                _ControlRecord.NameData = value;
            }
        }
        public ArrayControlRecordType RecordType
        {
            get
            {
                return _ControlRecord.RecordType;
            }
            set
            {
                // Set the new value or record type.
                _ControlRecord.RecordType = value;

                // If the new record type is CONSTANT, make sure that the data array 
                // is set to null.
                if (_ControlRecord.RecordType == ArrayControlRecordType.Constant)
                { _DataArray = null; }

            }
        }
        public T ConstantValue
        {
            get
            {
                return _ControlRecord.ConstantValue;
            }
            set
            {
                _ControlRecord.ConstantValue = value;
            }
        }
        public int FileUnit
        {
            get
            {
                return _ControlRecord.FileUnit;
            }
            set
            {
                _ControlRecord.FileUnit = value;
            }
        }
        public string Filename
        {
            get
            {
                return _ControlRecord.Filename;
            }
            set
            {
                _ControlRecord.Filename = value;
            }
        }
        public T ArrayMultiplier
        {
            get
            {
                return _ControlRecord.ArrayMultiplier;
            }
            set
            {
                _ControlRecord.ArrayMultiplier = value;
            }
        }
        public string InputFormat
        {
            get
            {
                return _ControlRecord.InputFormat;
            }
            set
            {
                _ControlRecord.InputFormat = value;
            }
        }
        public int PrintCode
        {
            get
            {
                return _ControlRecord.PrintCode;
            }
            set
            {
                _ControlRecord.PrintCode = value;
            }
        }
        public void SetAsConstant(T constantValue)
        {
            _ControlRecord.SetAsConstant(constantValue);
        }
        public void SetAsInternal(T arrayMultiplier, string inputFormat, int printCode)
        {
            _ControlRecord.SetAsInternal(arrayMultiplier, inputFormat, printCode);
        }
        public void SetAsExternal(int fileUnit, T arrayMultiplier, string inputFormat, int printCode)
        {
            _ControlRecord.SetAsExternal(fileUnit, arrayMultiplier, inputFormat, printCode);
        }
        public void SetAsOpenClose(string filename, T arrayMultiplier, string inputFormat, int printCode)
        {
            _ControlRecord.SetAsOpenClose(filename, arrayMultiplier, inputFormat, printCode);
        }
        public void SetControlRecordData(IArrayControlRecord<T> controlRecord)
        {
            _ControlRecord.SetControlRecordData(controlRecord);
        }

        #endregion

        #region Private Members
        /// <summary>
        /// 
        /// </summary>
        /// <param name="nameData"></param>
        /// <param name="textReader"></param>
        /// <param name="binaryReader"></param>
        /// <returns></returns>
        private Array2d<T> CreateProcessedArray(bool applyMultiplier, System.IO.StreamReader textReader, System.IO.BinaryReader binaryReader)
        {
            if (_ControlRecord.RecordType == ArrayControlRecordType.Constant)
            {
                Array2d<T> a = new Array2d<T>(RowCount, ColumnCount, _ControlRecord.ConstantValue);
                return a;
            }
            else if (_ControlRecord.RecordType == ArrayControlRecordType.Internal)
            {
                if (_DataArray == null)
                { return null; }
                else
                {
                    Array2d<T> a = _DataArray.GetCopy();
                    if (applyMultiplier)
                    {
                        a.Multiply(_ControlRecord.ArrayMultiplier);
                    }
                    return a;
                }
            }
            else if (_ControlRecord.RecordType == ArrayControlRecordType.OpenClose)
            {
                if (NameData != null)
                {
                    string pathname = NameData.GetFullFilename(_ControlRecord.Filename);
                    TextArrayIO<T> textIO = new TextArrayIO<T>();
                    Array2d<T> buffer = new Array2d<T>(RowCount, ColumnCount);
                    if (!textIO.Read(buffer, pathname))
                    {
                        throw new Exception("Error reading Open/Close file: " + pathname);
                    }
                    if (applyMultiplier)
                    {
                        buffer.Multiply(_ControlRecord.ArrayMultiplier);
                    }
                    return buffer;
                }
                else
                {
                    throw new ArgumentNullException("Cannot read Open/Close file because the Modflow name file information is missing.");
                }
                
            }
            else if (_ControlRecord.RecordType == ArrayControlRecordType.External)
            {
                if (binaryReader != null)
                {
                    IBinaryArrayFileIO<T> binIO = new BinaryArrayIO() as IBinaryArrayFileIO<T>;
                    Array2d<T> buffer = new Array2d<T>(RowCount, ColumnCount);
                    binIO.LoadFromStream(buffer, binaryReader);
                    if (applyMultiplier)
                    {
                        buffer.Multiply(_ControlRecord.ArrayMultiplier);
                    }
                    return buffer;
                }
                else if (textReader != null)
                {
                    TextArrayIO<T> textIO = new TextArrayIO<T>();
                    T[] buff = new T[RowCount * ColumnCount];
                    if (!textIO.Read(buff, textReader))
                    {
                        throw new Exception("Error reading array from external file.");
                    }
                    Array2d<T> buffer = new Array2d<T>(RowCount, ColumnCount);
                    if (applyMultiplier)
                    {
                        buffer.Multiply(_ControlRecord.ArrayMultiplier);
                    }
                    return buffer;
                }
                else
                {
                    if (NameData != null)
                    {
                        string pathname = NameData.GetFullFilename(this.FileUnit);
                        TextArrayIO<T> textIO = new TextArrayIO<T>();
                        Array2d<T> buffer = new Array2d<T>(RowCount, ColumnCount);
                        if (!textIO.Read(buffer, pathname))
                        {
                            throw new Exception("Error reading Open/Close file: " + pathname);
                        }
                        if (applyMultiplier)
                        {
                            buffer.Multiply(_ControlRecord.ArrayMultiplier);
                        }
                        return buffer;
                    }
                    else
                    {
                        throw new Exception("Could not process external file because the Modflow name file information is missing.");
                    }

                }

            }
            else
            {
                throw new Exception("Invalid array control record type.");
            }

        }
        #endregion

        #region IModflowDataArray<T> Members


        #endregion
    }
}
