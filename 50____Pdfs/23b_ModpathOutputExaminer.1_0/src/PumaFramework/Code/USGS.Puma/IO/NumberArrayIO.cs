using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.IO
{
    public class NumberArrayIO<T>
    {
        #region Text File IO
        /// <summary>
        /// Saves array values as a single line of comma-separated values.
        /// </summary>
        /// <param name="writer"></param>
        /// <param name="dataArray"></param>
        public static void SaveFile(System.IO.StreamWriter writer, T[] dataArray)
        {
            if (dataArray == null)
            { throw (new ArgumentNullException("dataArray")); }
            SaveFile(writer, dataArray, ',', dataArray.Length);
        }
        /// <summary>
        /// Saves array as a single line of delimited values.
        /// </summary>
        /// <param name="writer"></param>
        /// <param name="dataArray"></param>
        /// <param name="delimiter"></param>
        public static void SaveFile(System.IO.StreamWriter writer, T[] dataArray, char delimiter)
        {
            if (dataArray == null)
            {throw (new ArgumentNullException("dataArray"));}
            SaveFile(writer, dataArray, delimiter, dataArray.Length);
        }
        /// <summary>
        /// Saves array as a series of delimited value lines with the specified number of
        /// values per line.
        /// </summary>
        /// <param name="writer"></param>
        /// <param name="dataArray"></param>
        /// <param name="delimiter"></param>
        /// <param name="valuesPerLine"></param>
        public static void SaveFile(System.IO.StreamWriter writer, T[] dataArray, char delimiter, int valuesPerLine)
        {
            string[] output = null;

            try
            {
                bool validType = false;
                if (dataArray is int[]) validType = true;
                if (dataArray is float[]) validType = true;
                if (dataArray is double[]) validType = true;
                if (!validType) throw (new ArgumentException("The array must be a numeric data type."));
                if (writer == null) throw (new ArgumentNullException("writer"));

                int valsPerLine = valuesPerLine;
                if (valsPerLine > dataArray.Length) valsPerLine = dataArray.Length;

                output = NumberArrayIO<T>.ArrayToListOfStrings(dataArray, delimiter, "", 0, valsPerLine);
                if (output == null) throw (new Exception("Error saving array data to text file stream."));

                for (int i = 0; i < output.Length; i++)
                { writer.WriteLine(output[i]); }

                writer.Flush();

            }
            catch (Exception ex)
            {
                throw (new Exception("Error saving array data to text file stream."));
            }
        }
        #endregion

        #region Binary File IO
        public static void SaveFile(T[] values, System.IO.BinaryWriter writer)
        {
            IBinaryArrayFileIO<T> bio = (IBinaryArrayFileIO<T>)new BinaryArrayIO();
            bio.SaveInStream(values, writer);
        }
        public static void SaveFile(T[] values, System.IO.BinaryWriter writer, int startLocation)
        {
            IBinaryArrayFileIO<T> bio = (IBinaryArrayFileIO<T>)new BinaryArrayIO();
            bio.SaveInStream(values, writer, startLocation);
        }
        public static void SaveFile(T[] values, string filename)
        {
            IBinaryArrayFileIO<T> bio = (IBinaryArrayFileIO<T>)new BinaryArrayIO();
            bio.Save(values, filename);
        }
        public static void LoadFromStream(T[] values, System.IO.BinaryReader reader)
        {
            // Set startLocation to the beginning of the file and delegate to the overload 
            LoadFromStream(values, reader, 0);
        }
        public static void LoadFromStream(T[] values, System.IO.BinaryReader reader, int startLocation)
        {
            IBinaryArrayFileIO<T> bio = (IBinaryArrayFileIO<T>)new BinaryArrayIO();
            bio.LoadFromStream(values, reader, startLocation);
        }
        public static void Load(T[] values, string filename)
        {
            IBinaryArrayFileIO<T> bio = (IBinaryArrayFileIO<T>)new BinaryArrayIO();
            bio.Load(values, filename);
        }
        #endregion

        #region String IO
        /// <summary>
        /// Write values of an array as a comma-delimited string with the option of
        /// using run-length encoding to compress the result.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dataArray"></param>
        /// <param name="useRunLengthEncoding"></param>
        /// <returns></returns>
        public static string ArrayToString(T[] dataArray, bool useRunLengthEncoding)
        {
            if (useRunLengthEncoding)
            { 
                // delegate to private method.
                return ArrayToRleString(dataArray); 
            }
            else
            {   
                // delegate to the overload method using comma-delimiter, default starting position = 0, full length.
                return ArrayToString(dataArray, ',', "", 0, dataArray.Length);
            }
        }
        /// <summary>
        /// Write the values of an array to a delimited string.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dataArray"></param>
        /// <param name="delimiter"></param>
        /// <param name="dataFormat"></param>
        /// <returns></returns>
        public static string ArrayToString(T[] dataArray, char delimiter, string dataFormat)
        {
            if (dataArray == null)
            { throw (new ArgumentNullException("dataArray")); }
            // delegate to the overload method with default starting position = 0 and full length.
            return ArrayToString(dataArray, delimiter, dataFormat, 0, dataArray.Length);
        }
        /// <summary>
        /// Write the values of an array to a delimited string.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dataArray"></param>
        /// <param name="delimiter"></param>
        /// <param name="dataFormat"></param>
        /// <param name="startIndex"></param>
        /// <returns></returns>
        public static string ArrayToString(T[] dataArray, char delimiter, string dataFormat, int startIndex)
        {
            if (dataArray == null)
            { throw (new ArgumentNullException("dataArray")); }
            // delegate to the overload method with specified starting index and full length.
            return ArrayToString(dataArray, delimiter, dataFormat, startIndex, dataArray.Length);
        }
        /// <summary>
        /// Write the values of an array to a delimited string.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dataArray"></param>
        /// <param name="delimiter"></param>
        /// <param name="dataFormat"></param>
        /// <param name="startIndex"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static string ArrayToString(T[] dataArray, char delimiter, string dataFormat, int startIndex, int length)
        {
            // Make sure this is a numeric array that we know how to process.
            bool validType = false;
            if (dataArray is int[]) validType = true;
            if (dataArray is float[]) validType = true;
            if (dataArray is double[]) validType = true;
            if (!validType) throw (new ArgumentException("Data type is not a supported numeric type."));

            if (length == 0) return "";
            if (length < 0)
            { throw (new ArgumentException("length cannot be negative.")); }

            // Check to be sure that the starting index is valid.
            if (startIndex < 0) throw (new ArgumentException("Starting index cannot be negative.","startIndex"));
            if (startIndex > dataArray.Length - 1) throw (new ArgumentException("Starting index is greater than the length of the array.", "startIndex"));

            int endPosition = startIndex + length;
            if (endPosition > dataArray.Length) endPosition = dataArray.Length;
            if (endPosition <= startIndex) return "";

            int count = 0;
            string[] sList = new string[endPosition - startIndex];
            for (int i = startIndex; i<endPosition; i++)
            {
                sList[count] = dataArray[i].ToString();
                count++;
            }

            return string.Join(delimiter.ToString(), sList);

        }

        /// <summary>
        /// Write the values of an array to a set of strings with a specified
        /// number of values per string.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dataArray"></param>
        /// <param name="delimiter"></param>
        /// <param name="dataFormat"></param>
        /// <param name="startIndex"></param>
        /// <param name="valuesPerString"></param>
        /// <returns></returns>
        public static string[] ArrayToListOfStrings(T[] dataArray, char delimiter, string dataFormat, int startIndex, int valuesPerString)
        {
            try
            {
                // Make sure this is a numeric array that we know how to process.
                bool validType = false;
                if (dataArray is int[]) validType = true;
                if (dataArray is float[]) validType = true;
                if (dataArray is double[]) validType = true;
                if (!validType) throw (new ArgumentException("The array must be a numeric data type."));

                if (startIndex < 0) return null;

                int loc = startIndex;

                List<string> list = new List<string>();
                string[] sList = new string[dataArray.Length];
                for (int i = 0; i < sList.Length; i++)
                { sList[i] = dataArray[i].ToString(); }

                int length;
                string sDelimiter = delimiter.ToString();
                do
                {
                    length = valuesPerString;
                    if (loc + length >= sList.Length) length = sList.Length - loc;
                    list.Add(string.Join(sDelimiter,sList,loc,length));
                    loc += valuesPerString;
                }
                while (loc < sList.Length);

                return list.ToArray();


            }
            catch (Exception ex)
            {
                //System.Windows.Forms.MessageBox.Show(ex.Message);
                return null;
            }
        }

        /// <summary>
        /// Read values of an array from a delimited string. The string may be
        /// either a standard delimited string or may use run-length encoding.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rleString"></param>
        /// <returns></returns>
        public static T[] StringToArray(string s)
        {
            int i = 0;
            T val = default(T);

            try
            {
                if (string.IsNullOrEmpty(s)) return null;
                if (string.IsNullOrEmpty(s.Trim())) return null;

                int count = 0;
                string[] sBlock = null;
                //string[] sList = s.Split(new char[1] { ',' });
                char[] delimiters = new char[2] { ',', ' ' };
                string[] sList = s.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                List<T> data = new List<T>(sList.Length);
                IGenericNumberUtility<T> num = (IGenericNumberUtility<T>)new GenericNumberUtility();

                for (i = 0; i < sList.Length; i++)
                {
                    if (num.TryParse(sList[i], out val))
                    { data.Add(val); }
                    else
                    {
                        sBlock = sList[i].Split(new char[1] { ':' });
                        count = int.Parse(sBlock[0]);
                        val = num.Parse(sBlock[1]);
                        for (int j = 0; j < count; j++)
                        { data.Add(val); }
                    }
                }

                return data.ToArray();

            }
            catch (Exception ex)
            {
                return null;
            }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="s"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public static T[] StringToArray(string s, char delimiter)
        {
            try
            {
                if (string.IsNullOrEmpty(s)) return null;
                if (string.IsNullOrEmpty(s.Trim())) return null;

                int count = 0;
                string[] sList = s.Split(delimiter);
                List<T> data = new List<T>(sList.Length);
                IGenericNumberUtility<T> num = (IGenericNumberUtility<T>)new GenericNumberUtility();

                for (int i = 0; i < sList.Length; i++)
                { data.Add(num.Parse(sList[i])); }

                return data.ToArray();

            }
            catch (Exception ex)
            {
                return null;
            }

        }
        #endregion

        #region Private members
        /// <summary>
        /// Write the values of an array to a delimited string using run-length encoding compression.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dataArray"></param>
        /// <returns></returns>
        private static string ArrayToRleString(T[] dataArray)
        {
            IGenericNumberUtility<T> num = (IGenericNumberUtility<T>)new GenericNumberUtility();
            StringBuilder sb = null;
            string s = "";
            int startIndex = 0;
            int count = 0;

            T val = default(T);

            if (dataArray == null) return "";
            sb = new StringBuilder(2 * dataArray.Length);

            for (int i = 0; i < dataArray.Length; i++)
            {
                if (i == 0)
                {
                    startIndex = i;
                    val = dataArray[i];
                }

                if (!num.Equal(dataArray[i], val))
                {
                    count = i - startIndex;

                    if (count > 1)
                    { sb.Append(count).Append(":").Append(val).Append(","); }
                    else
                    { sb.Append(val.ToString()).Append(","); }

                    startIndex = i;
                    val = dataArray[i];

                }

                if (i == dataArray.Length - 1)
                {
                    count = i - startIndex + 1;

                    if (count > 1)
                    { sb.Append(count).Append(":").Append(val).Append(","); }
                    else
                    { sb.Append(val).Append(","); }
                }

            }

            return sb.ToString(0, sb.Length - 1);

        }
        #endregion

    }
}
