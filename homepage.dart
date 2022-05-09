import 'dart:io';
import 'dart:async';



import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';




class MyHomePage extends StatefulWidget {
   MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {


  File? _image;
  List? _outputs;
  final ImagePicker _picker = ImagePicker();

  //@override
 // void initState() {
  //  super.initState();
   // loadModel().then((value) {
   //   setState((){});
   // });
  //s}

  //@override
  //void dispose() async {
   // await Tflite.close();
   // super.dispose();
  //}


  //load your model

  Future loadModel() async {
    await Tflite.loadModel(
      model: "assets/model.tflite",
      labels: "assets/labels.txt",
    );
  }

  //run an image model
  Future runImageModel() async {
    //pick a random image
    final XFile? image = await _picker.pickImage(
        source: (Platform.isIOS ? ImageSource.gallery : ImageSource.camera),
        maxHeight: 224,
        maxWidth: 224);
    print(image!.path);
    //get prediction
    //labels are 1000 random english words for show purposes
    setState(() {
      _image = File(image.path);
      _outputs=null;
    });
  }

  Future galleryImageModel() async {
    //pick a random image
    final XFile? image = await _picker.pickImage(
        source: (ImageSource.gallery),
        maxHeight: 224,
        maxWidth: 224);
    print(image!.path);

    setState(() {
      _image = File(image.path);
      _outputs=null;
    });
  }

  classifyImage(File image) async {
    await Tflite.loadModel(
      model: "assets/model_fp16.tflite",
      labels: "assets/labels.txt",
    );

    Uint8List imageToByteListFloat32(
        img.Image image, int inputSize, double mean, double std) {
      int size=1 * inputSize * inputSize * 3;
      print(size);
      var convertedBytes = Float32List(size);
      print(convertedBytes.length);
      var buffer = Float32List.view(convertedBytes.buffer);
      print(convertedBytes.length);
      int pixelIndex = 0;
      for (var i = 0; i < inputSize; i++) {
        for (var j = 0; j < inputSize; j++) {

          var pixel = image.getPixelSafe(j, i);
          buffer[pixelIndex++] = (img.getRed(pixel))/225.0;
          buffer[pixelIndex++] = (img.getGreen(pixel)) / 225.0;
          buffer[pixelIndex++] = (img.getBlue(pixel)) / 225.0;
        }
      }
      print(convertedBytes.length);
      return convertedBytes.buffer.asUint8List();
    }

    var imageBytes =  image.readAsBytesSync().buffer;
    img.Image? oriImage = img.decodeJpg(imageBytes.asUint8List());
    img.Image resizedImage = img.copyResize(oriImage!, height: 224, width: 224);

    var output = await Tflite.runModelOnBinary(
         binary: imageToByteListFloat32(resizedImage, 224,224,224),// required
         numResults: 6,    // defaults to 5
         threshold: 0.05,  // defaults to 0.1
         asynch: true      // defaults to true
    );
    print(output);

    setState(() {
      _outputs = output;
    });
    await Tflite.close();
  }

  @override
  Widget build(BuildContext context) {
    return  Scaffold(
        appBar: AppBar(
          title: const Text('tflite Mobile Example'),
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null ? Image.network('https://code.recuweb.com/c/u/3a09f4cf991c32bd735fa06db67889e5/2018/08/wordpress-photo-gallery-plugins1.png', fit: BoxFit.cover,) : Image.file(_image!),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: <Widget>[
                IconButton(
                    onPressed: runImageModel,
                    icon: const Icon(Icons.camera_alt_outlined)),
                IconButton(
                    onPressed: galleryImageModel,
                    icon: const Icon(Icons.image_outlined))
              ],

            ),
            ElevatedButton(
              style: ButtonStyle(
                  backgroundColor: MaterialStateProperty.all(Colors.red[400])),
              child: const Text('Recognize image',
                style: TextStyle(color: Colors.white),),
              onPressed:  () {
                if(_image !=null)
                {
                  classifyImage(_image!);
                }else{
                  showDialog(
                    context: context,
                    barrierDismissible: true, // user must tap button!
                    builder: (BuildContext context) {
                      return AlertDialog(
                        title: const Text('alert'),
                        backgroundColor: Colors.white,
                        content: SingleChildScrollView(
                          child: ListBody(
                            children: const <Widget>[
                              Text('photo not selected'),
                            ],
                          ),
                        ),
                        actions: <Widget>[
                          TextButton(
                            child: const Text(
                              'ok',
                              style: TextStyle(color: Colors.black),
                            ),
                            onPressed: () {
                              Navigator.of(context).pop();
                            },
                          ),
                        ],
                      );
                    },
                  );
                }
                }
            ),
            Center(
              child: Visibility(
                visible: _outputs != null,
                child: _outputs !=null?Text('${_outputs![0]["label"]} - ${_outputs![0]["confidence"].toStringAsFixed(2)}(Confidence)',
                    style:const TextStyle( fontSize:20.0, fontWeight: FontWeight.bold)):const Text(''),
              ),
            ),
          ],
        ),
      );
  }
}