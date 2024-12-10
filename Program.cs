
using Tensorflow.NumPy;
using Tensorflow;
using Tensorflow.Keras;

const int ImgH = 100;
const int ImgW = 100;
const int NChannels = 1;//3;

Console.Write(ImgH);
Console.WriteLine(Directory.GetCurrentDirectory());

var basePath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"..","..","..", "Dataset"));
Console.WriteLine(basePath);
Console.WriteLine(AppContext.BaseDirectory);
var result = Directory.GetDirectories(basePath);

foreach (var x in result)
{
    Console.WriteLine(x);
}

List<int> yLabels = [];
List<string> xImagePath = [];
var fO = new FileOperation();

// Train Data
var trainPath = Path.Combine(basePath, "Train");
var testPath = Path.Combine(basePath, "Test");
var validatePath = Path.Combine(basePath, "Validation");

Dictionary<string, string[]> dict = fO.GetImagePath(trainPath);

var  xTrain = dict.ToDictionary(
    x=>x.Key, 
    x=>np.zeros((x.Value.Length, ImgH, ImgW, NChannels), dtype: TF_DataType.TF_FLOAT)
    );

foreach (var k in dict.Keys)
{
    fO.LoadImage(dict[k], xTrain[k],"Training");
}


//foreach (var x in dict)
//{
//    xTrain.Add(x.Key, np.zeros((x.Value.Length, ImgH, ImgW, NChannels), dtype: TF_DataType.TF_FLOAT));
//}

var p = new Preprocessing();

Console.WriteLine("TrainDB");
var train_db =p.image_dataset_from_directory(trainPath,
    validation_split: 0.0f,
    //label_mode:"binary", // int , cate
    label_mode:"categorical",
    subset: "training",
    //color_mode:"grayscale",
    seed: 111,
    image_size: (ImgH,ImgW)
    ,batch_size:64
    );
Console.WriteLine("TestDB");
var test_db = p.image_dataset_from_directory(testPath,
    validation_split: 0.0f,
    label_mode: "binary",
    subset: "training",
    seed: 222,
    //color_mode: "grayscale",
    image_size: (ImgH, ImgW)
    ,batch_size:16
);
Console.WriteLine("ValidDB");
var validate_db = p.image_dataset_from_directory(validatePath,
    //validation_split: 0.0f,
    label_mode: "binary",
    subset: "validation", 
    //color_mode: "grayscale",
    seed: 333,
    image_size: (ImgH, ImgW)
    ,batch_size:8
);

var m = new FaceMaskModel();

m.BuildModel(ImgH,ImgW,NChannels,2);
m.Train(train_db, validate_db);
m.Compile();
//Create Empty
// TF message coming from here
//var xTrain =  np.zeros((records.Count, ImgH, ImgW, NChannels), dtype: tf.float32); // TotalRecords * Height * width * Channel
//var yTrain = tf.one_hot(np.array(xLabels.ToArray(),dtype:tf.int64), depth: classCount);
//var yTrain = np.eye(classCount, dtype: tf.float32)[np.array(yLabels.ToArray(), tf.float32).reshape(-1)];
// Encode label to a one hot vector.

//var indexArray = np.array(xLabels.ToArray());  // N * xLabels.Total

//var one = yTrain[indexArray];

//indexArray = indexArray.reshape(-1);

//var one_hot_targets = np.eye(uniqueLabels.Length)[indexArray];
//var sh = one_hot_targets.shape;
//Load labels


Console.ReadLine();
