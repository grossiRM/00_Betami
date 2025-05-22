using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma;
using USGS.Puma.Core;
using USGS.Puma.IO;
using USGS.Puma.Utilities;

namespace USGS.Puma.Modflow
{
    public class ArrayControlRecord<T> : IArrayControlRecord<T>
    {
        #region Private Fields
        private IGenericNumberUtility<T> _Gnu = new GenericNumberUtility() as IGenericNumberUtility<T>;
        #endregion

        #region Constructors
        public ArrayControlRecord()
        {
            RecordType = ArrayControlRecordType.Constant;
        }
        public ArrayControlRecord(string controlRecordText) : this()
        {
            List<string> tokens = StringUtility.ParseAsFortranFreeFormat(controlRecordText, false);
            if (tokens != null)
            {
                if (tokens.Count > 0)
                {
                    string recType = tokens[0].ToUpper();
                    if (recType == "CONSTANT")
                    {
                        if (tokens.Count > 1)
                            SetAsConstant(_Gnu.Parse(tokens[1]));
                    }
                    if (recType == "INTERNAL")
                    {
                        T arrayMultiplier = default(T);
                        string inputFormat = "(FREE)";
                        int printCode = 0;
                        if (tokens.Count > 1) arrayMultiplier = _Gnu.Parse(tokens[1]);
                        if (tokens.Count > 2) inputFormat = tokens[2];
                        if (tokens.Count > 3) printCode = int.Parse(tokens[3]);
                        SetAsInternal(arrayMultiplier, inputFormat, printCode);
                    }
                    if (recType == "EXTERNAL")
                    {
                        int fileUnit = 0;
                        T arrayMultiplier = default(T);
                        string inputFormat = "(FREE)";
                        int printCode = 0;
                        if (tokens.Count > 1) fileUnit = int.Parse(tokens[1]);
                        if (tokens.Count > 2) arrayMultiplier = _Gnu.Parse(tokens[2]);
                        if (tokens.Count > 3) inputFormat = tokens[3];
                        if (tokens.Count > 4) printCode = int.Parse(tokens[4]);
                        SetAsExternal(fileUnit, arrayMultiplier, inputFormat, printCode);
                    }
                    if (recType == "OPEN/CLOSE")
                    {
                        string filename = "";
                        T arrayMultiplier = default(T);
                        string inputFormat = "(FREE)";
                        int printCode = 0;
                        if (tokens.Count > 1) filename = tokens[1];
                        if (tokens.Count > 2) arrayMultiplier = _Gnu.Parse(tokens[2]);
                        if (tokens.Count > 3) inputFormat = tokens[3];
                        if (tokens.Count > 4) printCode = int.Parse(tokens[4]);
                        SetAsOpenClose(filename, arrayMultiplier, inputFormat, printCode);
                    }
                }
            }
        }

        #endregion

        #region IArrayControlRecord<T> Members
        private ModflowNameData _NameData = null;
        /// <summary>
        /// 
        /// </summary>
        public ModflowNameData NameData
        {
            get
            {
                return _NameData;
            }
            set
            {
                _NameData = value;
            }
        }

        private ArrayControlRecordType _RecordType = ArrayControlRecordType.Constant;
        public ArrayControlRecordType RecordType
        {
            get
            {
                return _RecordType;
            }
            set
            {
                ArrayControlRecordType oldRecordType = _RecordType;
                _RecordType = value;

                if (_RecordType != oldRecordType)
                {
                    if (_RecordType == ArrayControlRecordType.Constant)
                    {
                        ConstantValue = default(T);
                        FileUnit = 0;
                        Filename = "";
                        ArrayMultiplier = _Gnu.Parse("1");
                        InputFormat = "(FREE)";
                        PrintCode = 0;
                    }
                    else if (_RecordType == ArrayControlRecordType.Internal)
                    {
                        ConstantValue = default(T);
                        FileUnit = 0;
                        Filename = "";
                        if (oldRecordType == ArrayControlRecordType.Constant)
                        {
                            ArrayMultiplier = _Gnu.Parse("1");
                            InputFormat = "(FREE)";
                            PrintCode = 0;
                        }
                    }
                    else if (_RecordType == ArrayControlRecordType.External)
                    {
                        ConstantValue = default(T);
                        FileUnit = 0;
                        Filename = "";
                        if (oldRecordType == ArrayControlRecordType.Constant)
                        {
                            ArrayMultiplier = _Gnu.Parse("1");
                            InputFormat = "(FREE)";
                            PrintCode = 0;
                        }
                    }
                    else if (_RecordType == ArrayControlRecordType.OpenClose)
                    {
                        ConstantValue = default(T);
                        FileUnit = 0;
                        Filename = "";
                        if (oldRecordType == ArrayControlRecordType.Constant)
                        {
                            ArrayMultiplier = _Gnu.Parse("1");
                            InputFormat = "(FREE)";
                            PrintCode = 0;
                        }
                    }

                }
            }
        }

        private T _ConstantValue = default(T);
        public T ConstantValue
        {
            get
            {
                return _ConstantValue;
            }
            set
            {
                _ConstantValue = value;
            }
        }

        private int _FileUnit = 0;
        public int FileUnit
        {
            get
            {
                return _FileUnit;
            }
            set
            {
                _FileUnit=value;
            }
        }

        private string _Filename = "";
        public string Filename
        {
            get
            {
                return _Filename;
            }
            set
            {
                _Filename=value;
            }
        }

        private T _ArrayMultiplier = default(T);
        public T ArrayMultiplier
        {
            get
            {
                return _ArrayMultiplier;
            }
            set
            {
                if (_Gnu.Equals(_Gnu.Parse("0")))
                    throw new Exception("Array multiplier cannot equal 0.");
                _ArrayMultiplier=value;
            }
        }

        private string _InputFormat = "";
        public string InputFormat
        {
            get
            {
                return _InputFormat;
            }
            set
            {
                _InputFormat=value;
            }
        }

        private int _PrintCode = 0;
        public int PrintCode
        {
            get
            {
                return _PrintCode;
            }
            set
            {
                _PrintCode=value;
            }
        }


        public void SetAsConstant(T constantValue)
        {
            RecordType = ArrayControlRecordType.Constant;
            ConstantValue = constantValue;
        }

        public void SetAsInternal(T arrayMultiplier, string inputFormat, int printCode)
        {
            RecordType = ArrayControlRecordType.Internal;
            ArrayMultiplier = arrayMultiplier;
            InputFormat = inputFormat;
            PrintCode = printCode;
        }

        public void SetAsExternal(int fileUnit, T arrayMultiplier, string inputFormat, int printCode)
        {
            RecordType = ArrayControlRecordType.External;
            FileUnit = fileUnit;
            ArrayMultiplier = arrayMultiplier;
            InputFormat = inputFormat;
            PrintCode = printCode;
        }

        public void SetAsOpenClose(string filename, T arrayMultiplier, string inputFormat, int printCode)
        {
            RecordType = ArrayControlRecordType.OpenClose;
            Filename = filename;
            ArrayMultiplier = arrayMultiplier;
            InputFormat = inputFormat;
            PrintCode = printCode;
        }

        public void SetControlRecordData(IArrayControlRecord<T> controlRecord)
        {
            switch (controlRecord.RecordType)
            {
                case ArrayControlRecordType.Constant:
                    SetAsConstant(controlRecord.ConstantValue);
                    break;
                case ArrayControlRecordType.Internal:
                    SetAsInternal(controlRecord.ArrayMultiplier, controlRecord.InputFormat, controlRecord.PrintCode);
                    break;
                case ArrayControlRecordType.External:
                    SetAsExternal(controlRecord.FileUnit, controlRecord.ArrayMultiplier, controlRecord.InputFormat, controlRecord.PrintCode);
                    break;
                case ArrayControlRecordType.OpenClose:
                    SetAsOpenClose(controlRecord.Filename, controlRecord.ArrayMultiplier, controlRecord.InputFormat, controlRecord.PrintCode);
                    break;
                default:
                    break;
            }
        }
        #endregion

        public int[] GetFixedFieldInfo()
        {
            throw new NotImplementedException();
        }

        private int[] ParseFortranFixedFormat(string fmt)
        {
            throw new NotImplementedException();

            //int tokenValue;
            //int[] returnValues = new int[4];
            //int[] values = new int[4];
            //string s = fmt.Trim().ToUpper();
            //if (string.IsNullOrEmpty(s)) { return; }
            //s = s.TrimStart('(');
            //s = s.TrimEnd(')');
            //if (string.IsNullOrEmpty(s)) {return;}

            //string[] tokens = s.Split('F', 'E', 'D', 'G', 'I');

            //if (tokens.Length < 2) { return; }

            //values[0] = int.Parse(tokens[0]);
            //tokens = s.Split('.');
            //if (int.TryParse(tokens[0], out tokenValue))
            //{
            //    values[1] = tokenValue;
            //    if (tokens.Length > 1)
            //    {
            //        if (int.TryParse(tokens[1], out tokenValue))
            //        {
            //            values[2] = tokenValue;
            //        }
            //        else
            //        { return; }
            //    }
            //}
            //else
            //{ return; }

            //if (s.Contains("I"))
            //{
            //    values[3] = 1;
            //}

            //for (int i = 0; i < 4; i++)
            //{
            //    returnValues[i] = values[i];
            //}

            //return;
        }

    }
}
