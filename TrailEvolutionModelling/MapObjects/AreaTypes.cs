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
        private const string DefaultAreaTypeName = "Lawn";
        private static readonly string Filename = "AreaTypes.xml";
        
        private static readonly AreaTypes Instance = Load();

        public static AreaType Default { get; private set; }

        public AreaType[] AreaTypeArray { get; set; }

        private Dictionary<string, AreaType> areaTypeDict;

        private AreaTypes() { }

        private void Init()
        {
            areaTypeDict = new Dictionary<string, AreaType>();
            foreach (var entry in AreaTypeArray)
            {
                areaTypeDict[entry.Name] = entry;
            }
            Default = areaTypeDict[DefaultAreaTypeName];
            AreaTypeArray = null;
        }

        public static AreaType GetByName(string name)
        {
            if (Instance.areaTypeDict.TryGetValue(name, out AreaType result))
                return result;

            throw new ArgumentException($"Unknown area type '{name}'");
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
            return new AreaTypes
            {
                AreaTypeArray = new AreaType[]
                {
                    new AreaType
                    {
                        Name = DefaultAreaTypeName,
                        DisplayedName = "Газон",
                        Style = new PolygonStyle(97, 170, 77) { PenStyle = PenStyle.LongDash },
                        Attributes = new AreaAttributes
                        {
                            IsWalkable = true,
                            IsTramplable = true,
                            Weight = 2.7f
                        }
                    },

                    new AreaType
                    {
                        Name = "PavedPath",
                        DisplayedName = "Пешеходная дорожка",
                        Style = new PolygonStyle(140, 140, 140) { PenStyle = PenStyle.LongDash },
                        Attributes = new AreaAttributes
                        {
                            IsWalkable = true,
                            IsTramplable = false,
                            Weight = 1
                        }
                    },

                    new AreaType
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
                    },

                    new AreaType
                    {
                        Name = "Vegetation",
                        DisplayedName = "Растительность",
                        Style = new PolygonStyle(41, 122, 47) { PenStyle = PenStyle.LongDash },
                        Attributes = new AreaAttributes
                        {
                            IsWalkable = true,
                            IsTramplable = true,
                            Weight = 1.5f
                        }
                    },

                    new AreaType
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
                    },

                    new AreaType
                    {
                        Name = "Building",
                        DisplayedName = "Здание",
                        Style = new PolygonStyle(50, 45, 45),
                        Attributes = AreaAttributes.Unwalkable
                    },

                    new AreaType
                    {
                        Name = "Fence",
                        DisplayedName = "Забор",
                        Style = new LineStyle(76, 76, 76),
                        Attributes = AreaAttributes.Unwalkable
                    },

                    new AreaType
                    {
                        Name = "Water",
                        DisplayedName = "Водоём",
                        Style = new PolygonStyle(116, 179, 224),
                        Attributes = AreaAttributes.Unwalkable
                    },

                    new AreaType
                    {
                        Name = "OtherUnwalkthroughable",
                        DisplayedName = "Прочее непроходимое препятствие",
                        Style = new PolygonStyle(12, 12, 12),
                        Attributes = AreaAttributes.Unwalkable
                    }
                }
            };
        }

        private static AreaTypes Load()
        {
            if (!File.Exists(Filename))
            {
                CreateDefaultFile();
            }

            try
            {
                using (var stream = File.OpenRead(Filename))
                {
                    var serializer = new XmlSerializer(typeof(AreaTypes));
                    var instance = (AreaTypes)serializer.Deserialize(stream);
                    instance.Init();
                    return instance;
                }
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
                
                return CreateDefaultAreaTypes();
            }
        }
    }
}
