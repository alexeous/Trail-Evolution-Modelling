using System.Reflection;
using Mapsui.Styles;

namespace TrailEvolutionModelling.Util
{
    static class BitmapResources
    {
        public static int GetBitmapIdForEmbeddedResourceRelative(string imagePath)
        {
            var assembly = Assembly.GetExecutingAssembly();
            string fullPath = assembly.GetName().Name + "." + imagePath;
            var names = assembly.GetManifestResourceNames();
            var image = assembly.GetManifestResourceStream(fullPath);
            return BitmapRegistry.Instance.Register(image);
        }
    }
}
