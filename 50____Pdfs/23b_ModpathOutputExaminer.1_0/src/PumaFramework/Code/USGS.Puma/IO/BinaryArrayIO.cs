using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using USGS.Puma.Core;
using USGS.Puma.Utilities;

namespace USGS.Puma.IO
{
    /// <summary>
    /// 
    /// </summary>
    public class BinaryArrayIO : IBinaryArrayFileIO<int>, IBinaryArrayFileIO<float>, IBinaryArrayFileIO<double>
    {

        #region IBinaryArrayFileIO<int> Members
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="writer"></param>
        public void SaveInStream(int[] buffer, System.IO.BinaryWriter writer)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (writer == null)
            { throw (new ArgumentNullException("The input BinaryWriter object is null.")); }

            for (int i = 0; i < buffer.Length; i++)
            { writer.Write(buffer[i]); }
          
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="writer"></param>
        /// <param name="startLocation"></param>
        public void SaveInStream(int[] buffer, System.IO.BinaryWriter writer, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (writer == null)
            { throw (new ArgumentNullException("The input BinaryWriter object is null.")); }
            if (startLocation >= writer.BaseStream.Length)
            { throw (new ArgumentException("Specified starting location is beyond the end of file."));}
            
            writer.BaseStream.Seek(startLocation, SeekOrigin.Begin);
            for (int i = 0; i < buffer.Length; i++)
            { writer.Write(buffer[i]); }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        public void Save(int[] buffer, string filename)
        {
            if (buffer == null)
            {throw (new ArgumentNullException("Input array is null."));}
            using (FileStream fs = new FileStream(filename, FileMode.Create,FileAccess.Write,FileShare.None))
            {
                using (BinaryWriter writer = new BinaryWriter(fs))
                {
                    for (int i=0; i<buffer.Length; i++)
                    { writer.Write(buffer[i]); }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public void LoadFromStream(int[] buffer, System.IO.BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }

            for (int i = 0; i < buffer.Length; i++)
            { buffer[i] = reader.ReadInt32(); }
 
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        /// <returns></returns>
        public void LoadFromStream(int[] buffer, System.IO.BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null."));}

            reader.BaseStream.Seek(startLocation, SeekOrigin.Begin);

            for (int i = 0; i < buffer.Length; i++)
            { buffer[i] = reader.ReadInt32(); }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        public void LoadFromStream(Array1d<int> buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            int[] buff = new int[buffer.ElementCount];

            LoadFromStream(buff, reader);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        public void LoadFromStream(Array1d<int> buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            int[] buff = new int[buffer.ElementCount];

            LoadFromStream(buff, reader, startLocation);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        public void LoadFromStream(Array2d<int> buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            int[] buff = new int[buffer.ElementCount];

            LoadFromStream(buff, reader);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        public void LoadFromStream(Array2d<int> buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            int[] buff = new int[buffer.ElementCount];

            LoadFromStream(buff, reader, startLocation);
            buffer.SetValues(buff);

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        /// <returns></returns>
        public void Load(int[] buffer, string filename)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if ( !System.IO.File.Exists(filename) )
            { throw (new ArgumentException("File does not exist: " + filename));}

            using (FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                using (BinaryReader reader = new BinaryReader(fs))
                {
                    for (int i = 0; i < buffer.Length; i++)
                    { buffer[i] = reader.ReadInt32(); }
                }
            }
        }

        #endregion

        #region IBinaryArrayFileIO<float> Members
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="writer"></param>
        public void SaveInStream(float[] buffer, BinaryWriter writer)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (writer == null)
            { throw (new ArgumentNullException("The input BinaryWriter object is null.")); }

            for (int i = 0; i < buffer.Length; i++)
            { writer.Write(buffer[i]); }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="writer"></param>
        /// <param name="startLocation"></param>
        public void SaveInStream(float[] buffer, BinaryWriter writer, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (writer == null)
            { throw (new ArgumentNullException("The input BinaryWriter object is null.")); }
            if (startLocation >= writer.BaseStream.Length)
            { throw (new ArgumentException("Specified starting location is beyond the end of file.")); }

            writer.BaseStream.Seek(startLocation, SeekOrigin.Begin);
            for (int i = 0; i < buffer.Length; i++)
            { writer.Write(buffer[i]); }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        public void Save(float[] buffer, string filename)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            using (FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                using (BinaryWriter writer = new BinaryWriter(fs))
                {
                    for (int i = 0; i < buffer.Length; i++)
                    { writer.Write(buffer[i]); }
                }
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public void LoadFromStream(float[] buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }

            for (int i = 0; i < buffer.Length; i++)
            { buffer[i] = reader.ReadSingle(); }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        /// <returns></returns>
        public void LoadFromStream(float[] buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }

            reader.BaseStream.Seek(startLocation, SeekOrigin.Begin);

            for (int i = 0; i < buffer.Length; i++)
            { buffer[i] = reader.ReadSingle(); }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        public void LoadFromStream(Array2d<float> buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            float[] buff = new float[buffer.ElementCount];

            LoadFromStream(buff, reader);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        public void LoadFromStream(Array2d<float> buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            float[] buff = new float[buffer.ElementCount];

            LoadFromStream(buff, reader, startLocation);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        public void LoadFromStream(Array1d<float> buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            float[] buff = new float[buffer.ElementCount];

            LoadFromStream(buff, reader);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        public void LoadFromStream(Array1d<float> buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            float[] buff = new float[buffer.ElementCount];

            LoadFromStream(buff, reader, startLocation);
            buffer.SetValues(buff);

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        /// <returns></returns>
        public void Load(float[] buffer, string filename)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (!System.IO.File.Exists(filename))
            { throw (new ArgumentException("File does not exist: " + filename)); }

            using (FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                using (BinaryReader reader = new BinaryReader(fs))
                {
                    for (int i = 0; i < buffer.Length; i++)
                    { buffer[i] = reader.ReadSingle(); }
                }
            }

        }

        #endregion

        #region IBinaryArrayFileIO<double> Members
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="writer"></param>
        public void SaveInStream(double[] buffer, BinaryWriter writer)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (writer == null)
            { throw (new ArgumentNullException("The input BinaryWriter object is null.")); }

            for (int i = 0; i < buffer.Length; i++)
            { writer.Write(buffer[i]); }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="writer"></param>
        /// <param name="startLocation"></param>
        public void SaveInStream(double[] buffer, BinaryWriter writer, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (writer == null)
            { throw (new ArgumentNullException("The input BinaryWriter object is null.")); }
            if (startLocation >= writer.BaseStream.Length)
            { throw (new ArgumentException("Specified starting location is beyond the end of file.")); }

            writer.BaseStream.Seek(startLocation, SeekOrigin.Begin);
            for (int i = 0; i < buffer.Length; i++)
            { writer.Write(buffer[i]); }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        public void Save(double[] buffer, string filename)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            using (FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                using (BinaryWriter writer = new BinaryWriter(fs))
                {
                    for (int i = 0; i < buffer.Length; i++)
                    { writer.Write(buffer[i]); }
                }
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public void LoadFromStream(double[] buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }

            for (int i = 0; i < buffer.Length; i++)
            { buffer[i] = reader.ReadDouble(); }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        /// <returns></returns>
        public void LoadFromStream(double[] buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }

            reader.BaseStream.Seek(startLocation, SeekOrigin.Begin);

            for (int i = 0; i < buffer.Length; i++)
            { buffer[i] = reader.ReadDouble(); }

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        public void LoadFromStream(Array1d<double> buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            double[] buff = new double[buffer.ElementCount];

            LoadFromStream(buff, reader);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        public void LoadFromStream(Array1d<double> buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            double[] buff = new double[buffer.ElementCount];

            LoadFromStream(buff, reader, startLocation);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        public void LoadFromStream(Array2d<double> buffer, BinaryReader reader)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            double[] buff = new double[buffer.ElementCount];

            LoadFromStream(buff, reader);
            buffer.SetValues(buff);

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="reader"></param>
        /// <param name="startLocation"></param>
        public void LoadFromStream(Array2d<double> buffer, BinaryReader reader, int startLocation)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (reader == null)
            { throw (new ArgumentNullException("The BinaryReader object is null.")); }
            double[] buff = new double[buffer.ElementCount];

            LoadFromStream(buff, reader, startLocation);
            buffer.SetValues(buff);

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="filename"></param>
        /// <returns></returns>
        public void Load(double[] buffer, string filename)
        {
            if (buffer == null)
            { throw (new ArgumentNullException("Input array is null.")); }
            if (!System.IO.File.Exists(filename))
            { throw (new ArgumentException("File does not exist: " + filename)); }

            using (FileStream fs = new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                using (BinaryReader reader = new BinaryReader(fs))
                {
                    for (int i = 0; i < buffer.Length; i++)
                    { buffer[i] = reader.ReadDouble(); }
                }
            }

        }

        #endregion

    }

}
