using System;
using System.Collections.Generic;
using System.Text;
using USGS.Puma.Core;

namespace USGS.Puma.Modflow
{
    public interface IArrayControlRecord<T>
    {
        ModflowNameData NameData { get; set; }
        ArrayControlRecordType RecordType { get; set; }
        T ConstantValue { get; set; }
        int FileUnit { get; set; }
        string Filename { get; set; }
        T ArrayMultiplier { get; set; }
        string InputFormat { get; set; }
        int PrintCode { get; set; }
        void SetAsConstant(T constantValue);
        void SetAsInternal(T arrayMultiplier, string inputFormat, int printCode);
        void SetAsExternal(int fileUnit, T arrayMultiplier, string inputFormat, int printCode);
        void SetAsOpenClose(string filename, T arrayMultiplier, string inputFormat, int printCode);
        void SetControlRecordData(IArrayControlRecord<T> controlRecord);
    }
    public interface IModflowDataArray<T>
    {
        int ArrayDimension { get; }
    }
    public interface IModflowDataArray1d<T> : IArrayControlRecord<T>
    {
        int ElementCount { get; }
        Array1d<T> DataArray { get; set; }
        Array1d<T> GetDataArrayCopy(bool applyMuliplier);
        Array1d<T> GetDataArrayCopy(System.IO.StreamReader reader, bool applyMultiplier);
        Array1d<T> GetDataArrayCopy(System.IO.BinaryReader reader, bool applyMultiplier);
    }
    public interface IModflowDataArray2d<T> : IArrayControlRecord<T>
    {
        int RowCount { get; }
        int ColumnCount { get; }
        Array2d<T> DataArray { get; set; }
        Array2d<T> GetDataArrayCopy(bool applyMultiplier);
        Array2d<T> GetDataArrayCopy(System.IO.StreamReader reader, bool applyMultiplier);
        Array2d<T> GetDataArrayCopy(System.IO.BinaryReader reader, bool applyMultiplier);
    }
    //public interface IStressPeriodDef<T>
    //{
    //    T PeriodLength { get; set; }
    //    int TimeStepCount { get; set; }
    //    T TimeStepMultiplier { get; set; }
    //    StressPeriodType PeriodType { get; set; }
    //}
    public interface IModflowNameData
    {
        void AddItem(int fileUnit, string fileType, string fileName);
        void AddItem(int fileUnit, string fileType, string fileName, string fileStatus);
        void AddItem(NameFileItem item);

        INameFileItem RemoveItem(int fileUnit);

        void Clear();

        bool FileUnitAvailable(int fileUnit);

        List<NameFileItem> GetItemsAsList();
    }
    public interface INameFileItem
    {
        string FileType { get; set; }
        int FileUnit { get; set; }
        string FileName { get; set; }
        string FileStatus { get; set; }
    }
}
