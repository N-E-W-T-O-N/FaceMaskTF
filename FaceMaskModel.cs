using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;

class FaceMaskModel
{
    private IKerasApi keras = Binding.tf.keras;

    private ILayersApi layers = Binding.tf.keras.layers;

    private IModel model { get; set; }



    //private void IntilizeModel()
    //{
    //    var inputs = layers.Input((150, 150, 3), name: "Input");
    //    var conv1 = layers.Conv2D(32, (3, 3), activation: "relu", padding: "same").Apply(inputs);
    //    var pool1 = layers.MaxPooling2D((2, 2)).Apply(conv1);
    //    var pooling = layers.MaxPooling2D(2, 2).Apply(pool1);


    //    var flat1 = layers.Flatten().Apply(pool2);


    //    var dense1 = layers.Dense(512, activation: "relu").Apply(concat);
    //    var dense2 = layers.Dense(128, activation: "relu").Apply(dense1);
    //    var dense3 = KerasApi.keras.layers.Dense(10, activation: "relu").Apply(dense2);
    //    //var output = layers.Softmax(-1).Apply(dense3);
    //    var output = layers.HardSigmoid().Apply(dense3);
    //    model = Binding.tf.keras.Model(inputs, output);

    //    model.summary();
    //}

    /// <summary>
    /// Build You model
    /// </summary>
    /// <param name="height">Height of Image</param>
    /// <param name="width">Width of Image</param>
    /// <param name="depth">Number Of Channels</param>
    /// <param name="classNumber">Total number of Classification labels</param>
    public void BuildModel(int height, int width, int depth, int classNumber)
    {
        var inputs = layers.Input(shape: (height, width, depth), name: "img");  //(32, 32, 3), name: "img");

        // convolutional layer
        var x = layers.Conv2D(8, (5, 5), padding: "same", activation: "relu").Apply(inputs);
        x = layers.BatchNormalization().Apply(x);
        x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);

        x = layers.Conv2D(16, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);

        x = layers.Conv2D(16, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);

        x = layers.Conv2D(32, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);

        x = layers.Conv2D(32, kernel_size: (3, 3), activation: "relu", padding: "same").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.MaxPooling2D(pool_size: (2, 2)).Apply(x);

        x = layers.Flatten().Apply(x);
        x = layers.Dense(128, activation: "relu").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.Dropout(0.5f).Apply(x);

        x = layers.Flatten().Apply(x);
        x = layers.Dense(128, activation: "relu").Apply(x);
        x = layers.BatchNormalization().Apply(x);
        x = layers.Dropout(0.5f).Apply(x);
        
        // output layer
        var outputs = layers.Dense(classNumber,"sigmoid").Apply(x);
        // build keras model
        model = keras.Model(inputs, outputs, name: "facemask_resnet");
    }

    /// <summary>
    /// Start Training of Image 
    /// </summary>
    /// <param name="xTrain"></param>
    /// <param name="yTrain"></param>
    /// <param name="classWeight"></param>
    /// <returns></returns>
    public ICallback Train(NDArray xTrain, NDArray yTrain, Dictionary<int, float> classWeight = null)
    {
        // training
        //model.fit(xTrain[new Slice(0, 2000)], yTrain[new Slice(0, 2000)],
        return model!.fit(xTrain, yTrain,
            batch_size: 64,
            epochs: 10,
            validation_split: 0.2f,
            class_weight: classWeight
            //,validation_data:new ValidationDataPack()
            );
    }

    public ICallback Train(IDatasetV2 xTrain, IDatasetV2 valid=null)
    {
        // training
        //model.fit(xTrain[new Slice(0, 2000)], yTrain[new Slice(0, 2000)],
        return model!.fit(xTrain,batch_size:64,epochs:10,validation_data:valid);

            //,validation_data:new ValidationDataPack()
    }

    /// <summary>
    /// Summary of Model trainned
    /// </summary>
    /// <exception cref="NullReferenceException"></exception>
    public void Summary()
    {
        if (model is null)
            throw new NullReferenceException("First call `BuildModel` Method to INITIALIZED the model object");

        model.summary();
    }

    /// <summary>
    /// Compile the Model
    /// </summary>
    public void Compile()
    {
        if (model is null)
            throw new NullReferenceException("First call `BuildModel` Method to INITIALIZED the model object");

        model!.compile(optimizer: keras.optimizers.RMSprop(1e-3f),
            loss: keras.losses.BinaryCrossentropy(from_logits: false),
            // keras.losses.CategoricalCrossentropy(from_logits: false), // SparseCategoricalCrossentropy(from_logits: true),
            metrics: [keras.metrics.BinaryAccuracy() ,keras.metrics.CategoricalAccuracy(), keras.metrics.CategoricalCrossentropy()]); //new[] { "acc" }); // //
    }

    /// <summary>
    /// Save trained Model Weight
    /// </summary>
    public void Save(string filePath) //"./toy_resnet_model"
    {
        if (model is null)
            throw new NullReferenceException("First call `BuildModel` Method to INITIALIZED the model object");

        // save the model
        model!.save(filePath, save_format: "tf");
    }

    /// <summary>
    /// Run prediction based on Trained model
    /// </summary>
    /// <param name="value"></param>
    /// <param name="verbose"></param>
    /// <returns>Tensor of size No. of Example * labelNumber</returns>
    public Tensor Predict(Tensor value, int verbose = 0)
    {
        // var c = confusion_matrix;
        var result = model.predict(value, verbose: verbose);
        return Binding.tf.arg_max(result, 1);
    }

    /// <summary>
    /// Load tf model 
    /// </summary>
    /// <param name="modelPath"></param>
    /// <exception cref="NullReferenceException"></exception>
    public void LoadMode(string modelPath)
    {
        if (String.IsNullOrEmpty(modelPath))
            throw new NullReferenceException("Please Provide the Path");

        model = Binding.tf.keras.models.load_model(modelPath);
        Console.WriteLine("Loding Model...");
        model.summary();
        Compile();
    }
}