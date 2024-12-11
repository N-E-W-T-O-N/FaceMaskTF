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
        var dimsExpander = Binding.tf.expand_dims(cast, 0,"expend");

        var resize = Binding.tf.constant(new int[] { imgH, imgW }, name: "resize");

        var bilinear = Binding.tf.image.resize_bilinear(dimsExpander, resize);//(dims_expander, resize);
        var sub = Binding.tf.subtract(bilinear, new float[] { 0 },name:"mean");
        var normalized = Binding.tf.divide(sub, new float[] { 255 },"SD");

        var sess = Binding.tf.Session(graph);
        return sess.run(normalized);

    }
    
    public (NDArray xDataArray, NDArray yDatArray) LoadFromDir(string path, string process)
    {
        List<string> imagesPath = [];
        List<int> labels = [];

        foreach ((int inx, string label) x in new[] { (0, "Mask"), (1, "Non MASK") })
        {
            //Find Total Files in Specific Label
            var files = Directory.GetFiles(Path.Combine(path, x.label));
            imagesPath.add(files);
            labels.add(Enumerable.Repeat(x.inx, files.Length));
        }

        NDArray xDataArray = np.zeros((imagesPath.Count, imgH, imgW, nChannels), dtype: TF_DataType.TF_FLOAT);
        LoadImage(imagesPath, xDataArray, process);
        NDArray yDatArray = np.array(labels.ToArray());

        return (xDataArray, yDatArray);
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

    public Dictionary<string, NDArray> LoadImage(string path,string process)
    {
        Dictionary<string, NDArray> dict = new ();

        foreach (var x in new[] { "Mask", "Non MASK" })
        {

            var files=  Directory.GetFiles(Path.Combine(path, x));
            NDArray nd = np.zeros((files.Length, imgH, imgW, nChannels), dtype: TF_DataType.TF_FLOAT);
            LoadImage(files, nd, process);

            dict.Add(x, nd);
        }

        return dict;


    }
    public void CreateImage(NDArray array, string name)
    {
        Binding.tf.image.encode_jpeg(array, name);
    }
}