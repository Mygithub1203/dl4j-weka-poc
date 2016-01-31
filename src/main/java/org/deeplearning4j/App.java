package org.deeplearning4j;

import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args ) throws Exception {

        int numClasses = 10;
        int width = 28;
        int height = 28;
        int channels = 1;
        
        
        /*DataSetIterator mnistIter = new MnistDataSetIterator(100,100, true);
        while(mnistIter.hasNext()) {
        	System.out.println(mnistIter.next());
        	break;
        }
        
        System.exit(0);
        */

        String labeledPath = "data";
        List<String> labels = new ArrayList<>();
        for(File f : new File(labeledPath).listFiles()) {
            String filename = f.getName();
            labels.add(filename);
        }
        System.out.println("labels: " + labels);

        ImageRecordReader reader = new ImageRecordReader(width, height, true, labels);
        reader.initialize(new FileSplit(new File(labeledPath)));
        DataSetIterator iter = new RecordReaderDataSetIterator(reader, width*height, labels.size());
        
        ArrayList<DataSet> images = new ArrayList<DataSet>();  
        while( iter.hasNext() ) {
        	DataSet d = iter.next();
        	//System.out.println( d.getLabels() );
        	d.divideBy(255);
        	images.add(d);
        }
        
        DataSet entireDataset = DataSet.merge(images);
        
        System.out.println("number of examples: " + entireDataset.numExamples());
        System.out.println("number of features: " + entireDataset.numInputs());
        System.out.println("number of classes: " + entireDataset.numOutcomes());
        
        /*
         java -Xmx14g weka.Run .LasagneNet \
        -S 1 \
        -layer ".Conv2DLayer -filter_size_x 5 -filter_size_y 5 -num_filters 16 -nonlinearity .Rectify" \
        -layer ".MaxPool2DLayer -pool_size_x 2 -pool_size_y 2" \
        -layer ".Conv2DLayer -filter_size_x 5 -filter_size_y 5 -num_filters 32 -nonlinearity .Rectify" \
        -layer ".MaxPool2DLayer -pool_size_x 2 -pool_size_y 2" \
        -layer ".DenseLayer -num_units 100 -nonlinearity .Rectify" \
        -loss ".CategoricalCrossEntropy -l1 0.0 -l2 0.0" \
        -update ".NesterovMomentum -momentum 0.9 -learning_rate 0.01" \
        -epochs 50 \
        -batch_iterator ".ImageBatchIterator -shuffle -batch_size 128 -width 40 -height 40 -prefix '/home/cjb60/Desktop/bugs_data/modified_40x40/canon-canoneos550d/${cls}' -rgb" \
        -eval_size ${eval_size} \
        -out_file `pwd`/conv1_${cls}.log \
        -t ../../data/${cls}_cleaned.arff ${eval_mode} > conv1_${cls}.txt
        */

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
        	.seed(0)
        	.iterations(100)
        	.gradientNormalization(GradientNormalization.None)
        	.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        	.learningRate(0.1)
        	.list(6)
        	.layer(0, new ConvolutionLayer.Builder(5, 5)
        			.name("conv1")
        			.stride(1,1)
        			.nIn(channels)
        			.nOut(16)
        			.activation("relu")
        			.weightInit(WeightInit.XAVIER)
        			.build()
    		)
    		.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 } )
    				.name("mp1")
    				.stride(2, 2)
    				.padding(0, 0)
    				.activation("linear")
    				.build()
    		)
        	.layer(2, new ConvolutionLayer.Builder(5, 5)
        			.name("conv2")
        			.stride(1,1)
        			.nOut(32)
        			.activation("relu")
        			.weightInit(WeightInit.XAVIER)
        			.build()
    		)
    		.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 } )
    				.name("mp2")
    				.stride(2, 2)
    				.padding(0, 0)
    				.activation("linear")
    				.build()
    		)
    		.layer(4, new DenseLayer.Builder()
    				.name("dense")
    				.nOut(100)
    				.weightInit(WeightInit.XAVIER)
    				.activation("relu")
    				.build()
    		)
    		.layer(5, new OutputLayer.Builder()
    				.name("out")
    				.nOut(numClasses)
    				.weightInit(WeightInit.XAVIER)
    				.activation("softmax")
    				.build()
    		)
    		.backprop(true)
    		.pretrain(false);
        
        new ConvolutionLayerSetup(builder, height, width, channels);
        
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        System.out.println("model configuration:");
        //System.out.println(model.conf().toYaml());
        
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));
        
        //Layer[] layers = model.getLayers();
        //for(int x = 0; x < layers.length; x++) {
        	//System.out.println("configuration of layer " + x + ":");
        	//System.out.println( layers[x].conf().toYaml() );  	
        	//System.out.println( "input shape: " + Arrays.toString(layers[x].params().shape()) );
        //}
        
        List<INDArray> activations = model.feedForward(entireDataset.get(0).getFeatureMatrix());
        for(int i = 0; i < activations.size(); i++) {
        	System.out.println(Arrays.toString(activations.get(i).shape()));
        }
        
    
        
        //model.fit(entireDataset);


    }
}
