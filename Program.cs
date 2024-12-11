using Tensorflow;
using Tensorflow.Util;

const int ImgH = 100;
const int ImgW = 100;
const int NChannels = 1; //3;

//Console.Write(ImgH);
Console.WriteLine(Directory.GetCurrentDirectory());

var basePath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"..","..","..", "Dataset"));
Console.WriteLine(basePath);
Console.WriteLine(AppContext.BaseDirectory);

//var result = Directory.GetDirectories(basePath);
//foreach (var x in result){Console.WriteLine(x);}

var fO = new FileOperation(ImgH,ImgW,NChannels);

// Train Data
var trainPath = Path.Combine(basePath, "Train");
var testPath = Path.Combine(basePath, "Test");
var validatePath = Path.Combine(basePath, "Validation");

Console.WriteLine("TrainDB");
var (xTrain,yTrain)  = fO.LoadFromDir(trainPath,"Training");
Console.WriteLine("TestDB");
var (xTest, yTest)   = fO.LoadFromDir(testPath, "Test");
Console.WriteLine("ValidDB");
var (xValid, yValid) = fO.LoadFromDir(validatePath, "Valid");

//var img = xTrain.numpy()[0];
//fO.CreateImage(img,Path.Combine(basePath,"Image.jpeg"));

var m = new FaceMaskModel();

// Binary Classifier Model ClassNumber as 1 as 
m.BuildModel(ImgH,ImgW,NChannels,1);
m.Compile();

Console.WriteLine("Training Started....");
m.Train(xTrain, yTrain,20,new ValidationDataPack((xValid,yValid)));

Console.WriteLine("Summary");
m.Summary();
Console.WriteLine();

var predict = m.Predict(xTest);

foreach (var p in predict)
{
    Console.WriteLine(p);
}

List<List<int>> matrix = new();

for (int ind = 0; ind < 2; ind++)
{
    matrix.add(Enumerable.Repeat(0, 2).ToList());
}
Console.WriteLine();

//for (int i = 0; i < 2; i++)
//{
//    try
//    {
//        int j = yTest.numpy()[i];
//        matrix[predict.numpy()[i]][j] += 1;
//    }
//    catch (Exception ex)
//    {
//        Console.WriteLine();
//    }
//}
Console.WriteLine();


Console.ReadLine();

