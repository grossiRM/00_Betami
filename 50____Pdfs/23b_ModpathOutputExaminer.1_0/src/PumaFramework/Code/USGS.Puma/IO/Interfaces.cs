using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;

namespace USGS.Puma.IO
{
    public interface IBinaryArrayFileIO<T>
    {
        void SaveInStream(T[] buffer, System.IO.BinaryWriter writer);
        void SaveInStream(T[] buffer, System.IO.BinaryWriter writer, int startLocation);
        void Save(T[] buffer, string filename);
        void LoadFromStream(T[] buffer, System.IO.BinaryReader reader);
        void LoadFromStream(T[] buffer, System.IO.BinaryReader reader, int startLocation);
        void LoadFromStream(Array1d<T> buffer, System.IO.BinaryReader reader);
        void LoadFromStream(Array1d<T> buffer, System.IO.BinaryReader reader, int startLocation);
        void LoadFromStream(Array2d<T> buffer, System.IO.BinaryReader reader);
        void LoadFromStream(Array2d<T> buffer, System.IO.BinaryReader reader, int startLocation);
        void Load(T[] buffer, string filename);
    }

}
