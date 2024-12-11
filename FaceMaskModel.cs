using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using Tensorflow.Util;

class FaceMaskModel
{
    private IKerasApi keras = Binding.tf.keras;

    private ILayersApi layers = Binding.tf.keras.layers;

    private IModel model { get; set; }

    private bool isCompiled = false;

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
    /// <param name="epoch"></param>
    /// <param name="validData"></param>
    /// <param name="classWeight"></param>
    /// <returns></returns>
    public ICallback Train(NDArray xTrain, NDArray yTrain,int epoch,
        ValidationDataPack validData =null ,    
            Dictionary<int, float> classWeight = null)
    {

        if (!isCompiled) throw new InvalidOperationException("Please call `Compile` method before Training");

        // training
        //model.fit(xTrain[new Slice(0, 2000)], yTrain[new Slice(0, 2000)],
        return model!.fit(xTrain, yTrain,
            batch_size: 64,
            epochs: epoch,
            validation_split: validData is null ? 0.2f : 0.0f,
            validation_data:validData,
            class_weight: classWeight
            );
    }

    public ICallback Train(IDatasetV2 xTrain, IDatasetV2 valid=null)
    {
        // training
        return model!.fit(xTrain,batch_size:64,epochs:10,validation_data:valid);
    }

    /// <summary>
    /// Summary of Model trained
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
        isCompiled = true;
        if (model is null)
            throw new NullReferenceException("First call `BuildModel` Method to INITIALIZED the model object");

        model!.compile(optimizer: keras.optimizers.RMSprop(1e-3f),
            loss: keras.losses.BinaryCrossentropy(from_logits: false,name:"binCross"),
            metrics: [keras.metrics.BinaryAccuracy(name:"binAcc") ,keras.metrics.Recall()
                //keras.metrics.SparseCategoricalAccuracy(name:"sparCateAcc"),
            ]); //new[] { "acc" }); // //
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
    public List<int> Predict(Tensor value, int verbose = 0)
    {
        // var c = confusion_matrix;
        var predict = model.predict(value, verbose: verbose);
        
        //var result =  Binding.tf.sigmoid(predict);
        //var l = predict.numpy().ToList();
        var z = predict.numpy().ToArray<float>();
        //var a = predict.numpy().ToMultiDimArray<float>();
        //foreach (var x in predict.numpy())
        //{
        //    Console.WriteLine(x[0].ToString());
        //}
        var r = z.Select(x => x > 0.5 ? 1 : 0).ToList();
        return r;
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
        Console.WriteLine("Loading Model...");
        model.summary();
        Compile();
    }
}