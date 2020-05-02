using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Xml.Serialization;
using Microsoft.Win32;

namespace TrailEvolutionModelling.Files
{
    class XmlSaverLoader<T>
    {
        public string Path { get; set; }

        private string filter;

        public XmlSaverLoader(string fileExtension)
        {
            filter = $"{fileExtension}|*.{fileExtension}";
        }

        public T Load()
        {
            var dialog = new OpenFileDialog { Filter = filter };
            if (dialog.ShowDialog() != true)
            {
                return default;
            }

            try
            {
                return ReadFromFile(dialog.FileName);
            }
            catch (Exception ex) when (
                ex is FileNotFoundException
                || ex is IOException
                || ex is InvalidOperationException
                || ex is ArgumentException
            )
            {
                MessageBox.Show("Ошибка при открытии файла: " + ex.ToString(), "Ошибка", 
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return default;
            }
        }

        public string Save(T obj)
        {
            if (Path == null)
                return SaveAs(obj);

            try
            {
                WriteToFile(Path, obj);
                return Path;
            }
            catch (Exception ex) when (
                ex is IOException
                || ex is InvalidOperationException
            )
            {
                MessageBox.Show("Ошибка при сохранении файла: " + ex.ToString(), "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return null;
            }
        }

        public string SaveAs(T obj)
        {
            var dialog = new SaveFileDialog { Filter = filter, FileName = Path };
            if (dialog.ShowDialog() != true)
                return null;

            try
            {
                WriteToFile(dialog.FileName, obj);
                Path = dialog.FileName;
                return Path;
            }
            catch (Exception ex) when (
                ex is IOException
                || ex is InvalidOperationException
            )
            {
                MessageBox.Show("Ошибка при сохранении файла: " + ex.ToString(), "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return null;
            }
        }


        private T ReadFromFile(string filename)
        {
            var serializer = new XmlSerializer(typeof(T));
            using (var stream = new FileStream(filename, FileMode.Open, FileAccess.Read))
            {
                var result = (T)serializer.Deserialize(stream);
                Path = filename;
                return result;
            }
        }

        private void WriteToFile(string filename, T obj)
        {
            var serializer = new XmlSerializer(typeof(T));
            using (var stream = new FileStream(filename, FileMode.Create, FileAccess.Write)) 
            {
                serializer.Serialize(stream, obj);
            }
        }
    }
}
