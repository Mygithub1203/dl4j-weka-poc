package org.deeplearning4j;

import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args ) throws Exception {

		/*
		 * GOAL: want to be able to create a DataSet consisting of all
		 * the images in the data/ folder, with all the associated
		 * labels so i can use it for a convolutional neural net
		 */

        int numClasses = 10;

        String labeledPath = "data";

        List<String> labels = new ArrayList<>();
        for(File f : new File(labeledPath).listFiles()) {
            String filename = f.getName();
            labels.add(filename);
        }

        System.out.println("labels: " + labels);

        ImageRecordReader reader = new ImageRecordReader(28, 28, true, labels);
        reader.initialize(new FileSplit(new File(labeledPath)));

        DataSetIterator iter = new RecordReaderDataSetIterator(reader, 784, labels.size());
        DataSet d = iter.next();

		/*
		 * input matrix is:
		 * -16,777,216.00, -16,777,216.00, -16,777,216.00, -16,777,216.00, -16,777,216.00, -16,777,216.00, -16,777,216.00, ...
		 * why is it doing this? i would have thought the numbers would be between 0 and 255 (or 0 and 1)...
		 */
        System.out.println(d.get(0));


    }
}
