# Classifying text with fastText in pySpark

## Background
We are working in a customer project where we need to classify hundreds of millions of messages based on the language. Python libraries such as *langdetect.py* fails to detect the language of the message. The reason behind poor performance for language detection libs in general is that they are trained on longer texts, and thus, they don't work in our special and rather challenging use case.

fastText [1] was chosen because it has shown excellent performance in text classification [2] and in language detection [3]. However, it is not trivial to run fastText in pySpark, thus, we wrote this guide.

## Requirements
To run the provided example, you need to have Apache Spark running either locally, e.g. on your laptop, or in cloud e.g. in AWS EMR. You can install spark with pySpark simply by writing

```
pip3 install pyspark
```
You may need to install Jupyter notebooks too
```
pip3 install jupyter
```
We also need fastText-python wrapper.
```
pip3 install fasttext
```


## Data

Before we can run a language classifier in Spark, we must train a classifier model. To train a model, we need to have known samples for each language we are interested. In this experiment, we used **X** dataset from **Y**. So, you need to download **XY** and store it in `data/` and decompress the file e.g. with command

```
tar -xvzf <file>
```

## Training a fastText classifier
### Train/test split data generation
Whenever we are building a model to make predictions/classifications, we need to evaluate its performance. Here is one example how we can divide our known data into train and test splits.

```py
# Define schema for the CSV file
schema = StructType([StructField("sentence_id", IntegerType(), True),
                     StructField("language_code", StringType(), True),
                     StructField("text", StringType(), True)])

# Read CSV into spark dataframe
spark_df = spark.read.csv('data/sentences.csv',
                          schema=schema, sep='\t')

# Split data into train and test
train_df, test_df = spark_df.randomSplit([0.8, 0.2], 42)
```

### Training a model
After splitting known data into train and test sets, we can start to train models. Python fastText-wrapper takes a filename and the name for the trained model file as inputs. We need to store training data into a file with the following format:

```
__label__<class> <text>
```

We can use the following script to store train and test samples in a file in the master node:

```py
# Storing train data
with open(TRAIN_FILE, 'w') as fp:
    for i, row in enumerate(train_df.toLocalIterator()):
        fp.write('__label__%s %s\n' % (row['language_code'],
                                       row['text']))
        if i % 1000 == 0:
            print(i)

# Storing test samples
with open(TEST_FILE, 'w') as fp:
    for i, row in enumerate(test_df.toLocalIterator()):
        fp.write('__label__%s %s\n' % (row['language_code'],
                                       row['text']))
        if i % 1000 == 0:
            print(i)
```

After saving training samples in a training file we can start training the model. It is can be done with fasttext.py lib as follows:

```py
import fasttext

TRAIN_FILE = 'data/fasttext_train.txt'
TEST_FILE = 'data/fasttext_test.txt'
MODEL_FILE = 'data/fasttext_language'

# We are training a supervised model with default parameters
# This the same as running fastText cli:
# ./fasttext supervised -input <trainfile> -model <modelfile>
model = fasttext.supervised(TRAIN_FILE,
                            MODEL_FILE)
```

## Classifying messages
To classify messages using the previously trained model we can write:
```py
import fasttext
model = fasttext.load_model(MODEL_FILE)
pred = model.predict(['Hello World!'])
```

To classify messages stored in a Spark DataFrame, we need to use Spark SQL User Defined Function (UDF). The UDF takes a function as an argument. Therefore we need to build a wrapper around fasttext classifier which includes a trained model (`model`), classification function (`model.predict`) and that returns only class label as string instead of a list of lists.

### Language classifier wrapper
Wrapper including a fastText model and a function which returns a predicted language for a given messages is following:

```py
# filename: fasttext_lang_classifier.py

# We need fasttext to load the model and make predictions
import fasttext

# Load model (loads when this library is being imported)
model = fasttext.load_model('data/model_fasttext.bin')

# This is the function we use in UDF to predict the language of a given msg
def predict_language(msg):
    pred = model.predict([msg])[0][0]
    pred = pred.replace('__label__', '')
    return pred
```

### Running language classifier
In order to use the custom fastText language wrapper library in Spark we use UDF as follows:

```py
# Import needed libraries
from pyspark.sql.functions import col, udf
# Import our custom fastText language classifier lib
import fasttext_lang_classifier
# Create a udf language classifier function
udf_predict_language = udf(fasttext_lang_classifier.predict_language)
```

We also don't want to forgot add our custom wrapper library and trained model in Spark context, thus we write:

```py
sc.addFile('data/model_fasttext.bin')
sc.addPyFile('fasttext_lang_classifier.py')
```
It is important to add files to Spark context especially when you are running Spark scripts in many worker nodes e.g. in AWS EMR. It will sync files all the worker nodes so that each worker can use the model to classify its partition of samples / rows.

Finally, we have a trained model, a UDF function with using our custom library and data loaded in a Spark DF. We can now predict languages for the messages in the DataFrame as follows:

```py
messages = messages.withColumn('predicted_lang',
                               udf_predict_language(col('text')))
```
We create a new column `predicted_lang` using `spark.sql.withColumn` where we store predicted language. We classify messages using our custom `udf_predict_language` function. It takes a column with messages to be classified as input i.e. `col('text')`. It returns a column consisting of predicted languages. This result will be stored in `predicted_lang`.

## References
1. [cite:fastextwebsite] fasText website
2. [cite:fasttextclasifier]
3. [cite:languageclassificationbench]
