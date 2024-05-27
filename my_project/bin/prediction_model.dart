import 'dart:io';

import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:ml_algo/ml_algo.dart';

void main(List<String> arguments) async {
  // Loading the dataset
  final data = await fromCsv('assets/data.csv');

  // // applying one-hot encoding on type
  final encoder = Encoder.oneHot(
    data,
    columnNames: ['type'],
  );
  final encodedData = encoder.process(data);

  // split into train and test
  final splits = splitData(encodedData, [0.8]);
  final trainData = splits[0];
  final testData = splits[1];

  // training the model
  final model = LinearRegressor(trainData, 'points',
      optimizerType: LinearOptimizerType.gradient,
      batchSize: trainData.rows.length,
      collectLearningData: true,
      iterationsLimit: 300,
      initialLearningRate: 1e-4);

  // Model Evaluation
  final error = model.assess(testData, MetricType.mape); // error 0.11

  print('mape error = $error');
  print('\n');
  print('iteration count = ${model.costPerIteration!.length}');
  print('\n');
  print('cost per iteration:');
  var i = 1;
  for (var cost in model.costPerIteration!) {
    print('iteration $i: $cost');
    i++;
  }

  await model.saveAsJson('score_system.json');

  final file = File('score_system.json');
  final encodedModel = await file.readAsString();
  final loadedModel = LinearRegressor.fromJson(encodedModel);
  // final unlabelledData = await fromJson('assets/data.json');
  final reportData = [
    ['distance', 'category', 'type', 'status'],
    [2.3, 3, 'Gas Leak', 1],
  ];

  final dataframe = DataFrame(reportData);

  final encodedUnlabelledData = encoder.process(dataframe);

  final prediction = loadedModel.predict(encodedUnlabelledData);
  print(prediction);
}
