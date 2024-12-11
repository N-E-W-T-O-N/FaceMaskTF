using Tensorflow;
using Tensorflow.NumPy;
class FileOperation(int imgH,int  imgW,int nChannels=1)
{
    public void LoadImage(IList<string> a, NDArray b, string process)
    {
        // Faster Approach
        Parallel.For(0, a.Count, (i) =>
        {
            try
            {
                var graph = Binding.tf.Graph().as_default();
                b[i] = ReadTensorFromImageFile(a[i], graph);
                Console.WriteLine($"Loading image: {i} {a[i]}...");
                Console.CursorLeft = 0;
                graph.Exit();
            }
            catch (Exception ex) { Console.WriteLine(ex.Message); }
        });

        Console.WriteLine();
        Console.WriteLine($"Loaded {a.Count} images for " + process);
    }

    private NDArray ReadTensorFromImageFile(string fileName, Graph graph)
    {
        var fileReader = Binding.tf.io.read_file(fileName, "file_reader");
        var decodeImage = Binding.tf.image.decode_jpeg(fileReader, channels: 3, name: "DecodeJpeg");
        //var decodeImage = tf.image.decode_image(fileReader, channels: 3, name: "DecodeImage");
        // Change Format to Float32 bit
        var cast = Binding.tf.cast(decodeImage, Binding.tf.float32, "cast");

        //resize required one extra dims
        var dims_expander = Binding.tf.expand_dims(cast, 0);

        var resize = Binding.tf.constant(new int[] { 32, 32 }, name: "resize");

        var bilinear = Binding.tf.image.resize_bilinear(dims_expander, resize);//(dims_expander, resize);
        var sub = Binding.tf.subtract(bilinear, new float[] { 0 });
        var normalized = Binding.tf.divide(sub, new float[] { 255 });

        var sess = Binding.tf.Session(graph);
        return sess.run(normalized);

    }

    public Dictionary<string, string[]> GetImagePath(string path)
    {
        var dict = new Dictionary<string, string[]>();

        foreach (var x in new[] { "Mask", "Non MASK" })
        {
            dict.Add(x, Directory.GetFiles(Path.Combine(path, x)));
        }

        return dict;
    }

    public void CreateImage(Dictionary<string, List<float>> history, string path)
    {

        foreach (var (name, data) in history)
        {

        }
    }
}