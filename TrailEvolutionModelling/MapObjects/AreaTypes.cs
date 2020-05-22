using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using System.Xml;
using System.Xml.Serialization;
using Mapsui.Styles;
using TrailEvolutionModelling.Styles;

namespace TrailEvolutionModelling.MapObjects
{
    [Serializable]
    public class AreaTypes
    {
        #region Area Types Properties

        public static AreaType[] All => new[]
        {
            Lawn, PavedPath, CarRoad, Vegetation, WalkthroughableFence,
            Building, Fence, Water, OtherUnwalkthroughable
        };

        public static AreaType Default => Lawn;

        public static AreaType Lawn { get; private set; } = new AreaType
        {
            Name = "Lawn",
            DisplayedName = "Газон",
            Style = new PolygonStyle(97, 170, 77) { PenStyle = PenStyle.LongDash },
            Attributes = new AreaAttributes
            {
                IsWalkable = true,
                IsTramplable = true,
                Weight = 2.7f
            }
        };

        public static AreaType PavedPath { get; private set; } = new AreaType
        {
            Name = "PavedPath",
            DisplayedName = "Пешеходная дорожка",
            Style = new PolygonStyle(170, 170, 170) { PenStyle = PenStyle.LongDash },
            Attributes = new AreaAttributes
            {
                IsWalkable = true,
                IsTramplable = false,
                Weight = 1
            }
        };

        public static AreaType CarRoad { get; private set; } = new AreaType
        {
            Name = "CarRoad",
            DisplayedName = "Проезжая часть",
            Style = new PolygonStyle(57, 74, 84) { PenStyle = PenStyle.LongDash },
            Attributes = new AreaAttributes
            {
                IsWalkable = true,
                IsTramplable = false,
                Weight = 1.2f
            }
        };

        public static AreaType Vegetation { get; private set; } = new AreaType
        {
            Name = "Vegetation",
            DisplayedName = "Растительность",
            Style = new PolygonStyle(41, 122, 47) { PenStyle = PenStyle.LongDash },
            Attributes = new AreaAttributes
            {
                IsWalkable = true,
                IsTramplable = false,
                Weight = 1.5f
            }
        };

        public static AreaType WalkthroughableFence { get; private set; } = new AreaType
        {
            Name = "WalkthroughableFence",
            DisplayedName = "Проходимый забор",
            Style = new LineStyle(63, 76, 96) { PenStyle = PenStyle.LongDash },
            Attributes = new AreaAttributes
            {
                IsWalkable = true,
                IsTramplable = false,
                Weight = 3.4f
            }
        };

        public static AreaType Building { get; private set; } = new AreaType
        {
            Name = "Building",
            DisplayedName = "Здание",
            Style = new PolygonStyle(50, 45, 45),
            Attributes = AreaAttributes.Unwalkable
        };

        public static AreaType Fence { get; private set; } = new AreaType
        {
            Name = "Fence",
            DisplayedName = "Непроходимый забор",
            Style = new LineStyle(76, 76, 76),
            Attributes = AreaAttributes.Unwalkable
        };

        public static AreaType Water { get; private set; } = new AreaType
        {
            Name = "Water",
            DisplayedName = "Водоём",
            Style = new PolygonStyle(116, 179, 224),
            Attributes = AreaAttributes.Unwalkable
        };

        public static AreaType OtherUnwalkthroughable { get; private set; } = new AreaType
        {
            Name = "OtherUnwalkthroughable",
            DisplayedName = "Другое непроходимое препятствие",
            Style = new PolygonStyle(12, 12, 12),
            Attributes = AreaAttributes.Unwalkable
        };

        #endregion

        private static Dictionary<string, AreaType> areaTypeDict;
        private static readonly string Filename = "AreaTypes.xml";

        public AreaType[] AreaTypeArray { get; set; }
        private AreaTypes() { }


        static AreaTypes()
        {
            InitDictionaryDefault();

            try
            {
                AreaTypes instance = Load();
                OverrideDefaultAreaTypes(instance);
            }
            catch (Exception ex) when (
                  ex is XmlException
               || ex is InvalidOperationException
               || ex is ArgumentException
            )
            {
                MessageBox.Show($"Не удалось загрузить свойства областей из {Filename}. " +
                        "Будут использованы значения по умолчанию. Техническая информация:\n" + ex.ToString(),
                        "Предупреждение",
                        MessageBoxButton.OK,
                        MessageBoxImage.Warning);
            }
        }

        private static void InitDictionaryDefault()
        {
            areaTypeDict = new Dictionary<string, AreaType>();
            foreach (var areaType in All)
            {
                areaTypeDict[areaType.Name] = areaType;
            }
        }

        private static void OverrideDefaultAreaTypes(AreaTypes instance)
        {
            foreach (var areaType in instance.AreaTypeArray)
            {
                if (areaTypeDict.TryGetValue(areaType.Name, out AreaType target))
                {
                    target.CopyValuesFrom(areaType);
                }
            }
        }

        public static AreaType GetByName(string name)
        {
            if (areaTypeDict.TryGetValue(name, out AreaType result))
                return result;

            throw new ArgumentException($"Unknown area type '{name}'");
        }

        private static AreaTypes Load()
        {
            if (!File.Exists(Filename))
            {
                CreateDefaultFile();
            }

            using (var stream = File.OpenRead(Filename))
            {
                var serializer = new XmlSerializer(typeof(AreaTypes));
                var instance = (AreaTypes)serializer.Deserialize(stream);
                return instance;
            }
        }

        private static AreaTypes CreateDefaultFile()
        {
            var instance = CreateDefaultAreaTypes();
            using (var stream = new FileStream(Filename, FileMode.Create, FileAccess.Write))
            {
                var serializer = new XmlSerializer(typeof(AreaTypes));
                serializer.Serialize(stream, instance);
            }
            return instance;
        }

        private static AreaTypes CreateDefaultAreaTypes()
        {
            return new AreaTypes { AreaTypeArray = All };
        }
    }
}
