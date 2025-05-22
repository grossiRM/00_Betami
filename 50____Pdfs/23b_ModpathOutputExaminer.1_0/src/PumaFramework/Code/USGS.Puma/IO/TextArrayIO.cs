using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.IO
{
    public class TextArrayIO<T>
    {
        #region Private Fields
        private IGenericNumberUtility<T> _GNU = new GenericNumberUtility() as IGenericNumberUtility<T>;
        #endregion

        #region Public Methods

        #region Write Methods
        public void Write(Array1d<T> buffer, string filename, char delimiter)
        {
            using (System.IO.StreamWriter writer = new System.IO.StreamWriter(filename))
            { Write(buffer, writer, delimiter); }
        }
        public void Write(Array1d<T> buffer, System.IO.StreamWriter writer, char delimiter)
        {
            for (int i = 1; i <= buffer.ElementCount; i++)
            {
                writer.Write(buffer[i]);
                if (i < buffer.ElementCount)
                    writer.Write(delimiter);
                else
                    writer.Write(Environment.NewLine);
            }
        }
        public void Write(Array1d<T> buffer, string filename,char delimiter,int valuesPerLine)
        {
            using (System.IO.StreamWriter writer = new System.IO.StreamWriter(filename))
            {
                T[] buf;
                buf = buffer.ToArray();
                NumberArrayIO<T>.SaveFile(writer, buf, delimiter, valuesPerLine);
            }

        }
        public void Write(Array1d<T> buffer, System.IO.StreamWriter writer)
        {
            T[] buf;
            buf = buffer.ToArray();
            NumberArrayIO<T>.SaveFile(writer, buf);
        }
        public void Write(Array2d<T> buffer, string filename, char delimiter)
        {
            using (System.IO.StreamWriter writer = new System.IO.StreamWriter(filename))
            { Write(buffer, writer, delimiter); }
        }
        public void Write(Array2d<T> buffer, System.IO.StreamWriter writer, char delimiter)
        {
            for (int row = 1; row <= buffer.RowCount; row++)
            {
                for (int column = 1; column <= buffer.ColumnCount; column++)
                {
                    writer.Write(buffer[row, column]);
                    if (column < buffer.ColumnCount)
                        writer.Write(delimiter);
                    else
                        writer.Write(Environment.NewLine);
                }
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="delimiter"></param>
        public void Write(Array2d<T> buffer, string filename, char delimiter, int valuesPerLine)
        {
            using (System.IO.StreamWriter writer = new System.IO.StreamWriter(filename))
            {
                T[] buf;

                for (int row = 1; row <= buffer.RowCount; row++)
                {
                    buf = buffer.ToRowArray(row);
                    NumberArrayIO<T>.SaveFile(writer, buf, delimiter, valuesPerLine);
                }
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="writer"></param>
        public void Write(Array2d<T> buffer, System.IO.StreamWriter writer)
        {
            T[] buf;
            for (int row = 1; row <= buffer.RowCount; row++)
            {
                buf = buffer.ToRowArray(row);
                NumberArrayIO<T>.SaveFile(writer, buf);
            }
        }
        #endregion

        #region Read Methods
        /// <summary>
        /// 
        /// </summary>
        /// <param name="columnCount"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        private T[] ReadRowAsFreeFormat(int columnCount, System.IO.StreamReader reader)
        {
            int count = 0;
            string line = null;
            T[] a = new T[columnCount];
            List<string> tokens = null;
            string token = null;
            
            while (count < columnCount)
            {
                line = reader.ReadLine();
                tokens = StringUtility.ParseAsFortranFreeFormat(line, false);
                for (int i = 0; i < tokens.Count; i++)
                {
                    a[count] = _GNU.Parse(tokens[i]);
                    count++;
                    if (count == columnCount) break;
                }
            }

            return a;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="columnCount"></param>
        /// <param name="reader"></param>
        /// <param name="valuesPerLine"></param>
        /// <param name="fieldWidth"></param>
        /// <returns></returns>
        private T[] ReadRowAsFixedFormat(int columnCount, System.IO.StreamReader reader, int valuesPerLine, int fieldWidth)
        {
            throw new NotImplementedException("Fixed format read not implemented.");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        /// <returns></returns>
        public bool Read(T[] buffer, string filename)
        {
            bool result = false;
            if (buffer == null) return result;

            using (System.IO.StreamReader reader = new System.IO.StreamReader(filename))
            {
                result = Read(buffer, reader);
            }

            return result;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        /// <returns></returns>
        public bool Read(Array1d<T> buffer, string filename)
        {
            bool result = false;
            if (buffer == null) return result;

            using (System.IO.StreamReader reader = new System.IO.StreamReader(filename))
            {
                result = Read(buffer, reader);
            }

            return result;
        
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        /// <param name="inputFormat"></param>
        /// <returns></returns>
        public bool Read(Array1d<T> buffer, string filename, string inputFormat)
        {
            string sFormat = inputFormat.Trim().ToUpper();
            if (string.IsNullOrEmpty(sFormat)) sFormat = "(FREE)";
            if (sFormat == "(FREE)")
            {
                return Read(buffer, filename);
            }
            else
            {
                throw new Exception("The specified input format is not supported.");
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public bool Read(T[] buffer, System.IO.StreamReader reader)
        {
            bool result = false;
            if (buffer == null) return result;
            int count = 0;
            string line = null;
            List<string> tokens = null;
            string token = null;

            while (count < buffer.Length)
            {
                line = reader.ReadLine();
                tokens = StringUtility.ParseAsFortranFreeFormat(line, false);
                for (int i = 0; i < tokens.Count; i++)
                {
                    count++;

                    buffer[count-1] = _GNU.Parse(tokens[i]);
                    if (count == buffer.Length) break;
                }
            }

            return true;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public bool Read(Array1d<T> buffer, System.IO.StreamReader reader)
        {
            bool result = false;
            if (buffer == null) return false;

            T[] rowBuffer = null;
            rowBuffer = ReadRowAsFreeFormat(buffer.ElementCount, reader);
            for (int i = 0; i < buffer.ElementCount; i++)
            { buffer[i + 1] = rowBuffer[i]; }
            return true;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="inputFormat"></param>
        /// <returns></returns>
        public bool Read(Array1d<T> buffer, System.IO.StreamReader reader, string inputFormat)
        {
            string sFormat = inputFormat.Trim().ToUpper();
            if (string.IsNullOrEmpty(sFormat)) sFormat = "(FREE)";
            if (sFormat == "(FREE)")
            {
                return Read(buffer, reader);
            }
            else
            {
                throw new Exception("The specified input format is not supported.");
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="optionList"></param>
        public bool Read(Array2d<T> buffer, string filename)
        {
            bool result = false;

            if (buffer == null) return false;

            using (System.IO.StreamReader reader = new System.IO.StreamReader(filename))
            {
               result  = Read(buffer, reader);
            }

            return result;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        /// <param name="inputFormat"></param>
        /// <returns></returns>
        public bool Read(Array2d<T> buffer, string filename, string inputFormat)
        {
            bool result = false;

            if (buffer == null) return false;

            using (System.IO.StreamReader reader = new System.IO.StreamReader(filename))
            {
                result = Read(buffer, reader, inputFormat);
            }

            return result;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="reader"></param>
        /// <param name="optionList"></param>
        public bool Read(Array2d<T> buffer, System.IO.StreamReader reader)
        {
            bool result = false;
            if (buffer == null) return false;

            T[] rowBuffer = null;
            for (int row = 1; row <= buffer.RowCount; row++)
            {
                rowBuffer = ReadRowAsFreeFormat(buffer.ColumnCount, reader);
                for (int i = 0; i < buffer.ColumnCount; i++)
                { buffer[row, i + 1] = rowBuffer[i]; }
            }
            return true;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="inputFormat"></param>
        /// <returns></returns>
        public bool Read(Array2d<T> buffer, System.IO.StreamReader reader, string inputFormat)
        {
            string sFormat = inputFormat.Trim().ToUpper();
            if (string.IsNullOrEmpty(sFormat)) sFormat = "(FREE)";
            if (sFormat == "(FREE)")
            {
                return Read(buffer, reader);
            }
            else
            {
                throw new Exception("The specified input format is not supported.");
            }
        }
        #endregion

        #endregion

    }
}
