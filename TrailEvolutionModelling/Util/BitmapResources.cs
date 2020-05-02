using System;
using System.Reflection;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Mapsui.Styles;

namespace TrailEvolutionModelling.Util
{
    static class BitmapResources
    {
        public static int GetBitmapIdForEmbeddedResourceRelative(string filename)
        {
            var assembly = Assembly.GetExecutingAssembly();
            string fullPath = assembly.GetName().Name + ".Resources." + filename;
            var image = assembly.GetManifestResourceStream(fullPath);
            return BitmapRegistry.Instance.Register(image);
        }

        public static Image LoadImage(string filename)
        {
            string uriStr = "pack://application:,,,/Resources/" + filename;
            return new Image { Source = new BitmapImage(new Uri(uriStr)) };
        }
    }
}
