/**
 * MainActivity.java
 * @author Jonathan Dowdall
 * @since 06-15-2016
 */
package com.example.jonny.updateweights;

import android.content.DialogInterface;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.InputType;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.IntBuffer;
import java.util.Random;

import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class MainActivity extends AppCompatActivity {

    /** Maximum size of W and input vectors */
    public int mSizeW;

    /** Kernel filename for OpenCL to load */
    public String mKernelName = "UpdateWeights.cl";

    /** Input array used for updating weights */
    public float mInput[];

    /** Array storing input averages for CPU computation */
    public float mCpuW[];

    /** Array storing input averages for CPU computation */
    public float mGpuW[];


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        /**
         * Initialize OpenCL
         */
        final Button buttonInitializeCl = (Button) findViewById(R.id.initCl);
        buttonInitializeCl.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v) {
                // Load kernel from file
                copyFile(mKernelName);

                // Initialize OpenCL api and connect to GPU
                // Convert result from integer to GpuProperty enum type for clarity
                GpuProperty gpuResult = GpuProperty.values()[initOpenCl(mKernelName)];

                // Report GPU properties
                String gpuInfo = "Initialization failed.";
                if (gpuResult == GpuProperty.UNAVAILABLE) gpuInfo = "Unavailable";
                if (gpuResult == GpuProperty.DISCRETE) gpuInfo = "Discrete";
                if (gpuResult == GpuProperty.INTEGRATED) gpuInfo = "Integrated";
                String output = "OpenCL Initialization Complete. \nGPU: " + gpuInfo;
                ((TextView) findViewById(R.id.result)).setText(output);
            }
        });

        final Button buttonInitializeW = (Button) findViewById(R.id.initW);
        buttonInitializeW.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                // Initialize W on GPU and set size of vector
                mSizeW = initW();

                // Allocate space on CPU in order to check GPU correctness
                // once computation is complete
                mGpuW = new float[mSizeW];

                // Initialize input and W on CPU
                mInput = new float[mSizeW];

                mCpuW = new float[mSizeW];

                // Report size of vectors
                String output = "Array Initialization Complete.\n" +
                        Integer.toString(mSizeW) + " elements";
                ((TextView) findViewById(R.id.result)).setText(output);
            }
        });

        final Button buttonRun = (Button) findViewById(R.id.run);
        buttonRun.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                // Receive user input for final value of T
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("Time");
                final EditText timeInput = new EditText(MainActivity.this);
                // Specify the type of input expected
                timeInput.setInputType(InputType.TYPE_CLASS_NUMBER);
                builder.setView(timeInput);

                // Set up the buttons
                builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        int finalTime = Integer.parseInt(timeInput.getText().toString());
                        double cpuTime = 0;
                        double gpuTime = 0;
                        long cpuStart, cpuEnd, gpuStart, gpuEnd;
                        for (int time = 1; time <= finalTime; ++time ) {
                            // Create mock input using random array of floats
                            randomizeFloats(mInput);

                            // Update CPU weights and keep track of runtime
                            cpuStart = System.nanoTime();
                            updateWeights(mCpuW, mInput, time);
                            cpuEnd = System.nanoTime();
                            cpuTime += (double)(cpuEnd - cpuStart) / 1000000;

                            // Update GPU weights and keep track of runtime
                            gpuStart = System.nanoTime();
                            updateWeights(mInput, time);
                            gpuEnd = System.nanoTime();
                            gpuTime += (double)(gpuEnd - gpuStart) / 1000000;
                        }

                        // Retreive GPU w array in order to check results
                        mGpuW = getGpuW();

                        double relativeError = computeRelativeError(mCpuW, mGpuW);

                        // Output results
                        String result = "Results:\n";
                        result += "CPU Runtime: " + Double.toString(cpuTime) + " ms\n";
                        result += "GPU Runtime: " + Double.toString(gpuTime) + " ms\n";
                        result += "\nRuntime reduction: " +
                                Double.toString((double)(1 - gpuTime/cpuTime) * 100) + "%\n";

                        result += "\nGPU relative error to CPU: " +
                                Double.toString(relativeError*100) + "%";

                        // Print first 10 weights in each vector
                        for (int i = 0; i < 10; ++i)
                        {
                            result += String.format("\n\nmCpuW[%d]: %f", i, mCpuW[i]);
                            result += String.format("\nmGpuW[%d]: %f", i, mGpuW[i]);
                        }
                        ((TextView) findViewById(R.id.result)).setText(result);
                    }
                });
                builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();
                    }
                });
                AlertDialog alert = builder.create();
                alert.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE);
                alert.show();
            }
        });
    }


    @Override
    public void onStart() {
        super.onStart();

    }

    @Override
    public void onStop() {
        super.onStop();

    }

    /**
     * Used to define and pass data types to native functions.
     */
    enum NativeType {
        JBoolean,
        JByte,
        JChar,
        JShort,
        JInt,
        JLong,
        JFloat,
        JDouble,
    }

    /**
     * Used to return properties of GPU from OpenCL initialization.
     */
    enum GpuProperty {
        /**
         * No GPU is found on platform.
         */
        UNAVAILABLE,

        /**
         * GPU is found with dedicated memory (discrete graphics card
         */
        DISCRETE,

        /**
         * GPU is found with host-unified memory (integrated graphics card)
         */
        INTEGRATED
    }


    /**
     * A native method that creates the OpenCL context and connects to a GPU device.
     *
     * @param kernelName The name of the kernel to load upon initialization
     * @return           An int to be converted to an enum indicating availibility of GPU.
     *                   If available, indicates discrete or integrated graphics.
     * @see GpuProperty
     */
    public native int initOpenCl(String kernelName);

    /**
     * A native method that initializes an array W on the GPU device.
     *
     * @return An int indicating the size of the array W
     */
    public native int initW();

    /**
     * A native method that updates all input averages via cpu and gpu.
     *
     * @param time  How many iterations to generate input and update weights
     */
    public native int updateWeights(float[] input, int time);

    /**
     * @return The array W that was used in the GPU computation.
     */
    public native float[] getGpuW();

    /**
	 * Loads the kernel into the app_execdir.
     *
     * @param f the filename of the kernel
	 */
    private void copyFile(final String f)
    {
        InputStream in;
        try {
            in = getAssets().open(f);
            final File of = new File(getDir("execdir",MODE_PRIVATE), f);

            final OutputStream out = new FileOutputStream(of);

            final byte b[] = new byte[65535];
            int sz = 0;
            while ((sz = in.read(b)) > 0) {
                out.write(b, 0, sz);
            }
            in.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Fills array with random float values between 1 and -1
     *
     * @param input the array that is being randomized
     */
    private void randomizeFloats(float[] input)
    {
        Random rand = new Random();

        for (int i = 0; i < mSizeW; ++i)
        {
            input[i] = rand.nextFloat();
        }
    }

    /**
     * Updates weights with new input in order to maintain input average
     *
     * @param weights Array of floats containing current input average
     * @param   input Array of floats representing new input to weights
     * @param    time Current iteration of input for proper average maintenance
     */
    private void updateWeights(float[] weights, float input[], float time)
    {
        for (int i = 0; i < weights.length; ++i) {
            weights[i] = ((float)(time - 1) / time * mCpuW[i]) + ((float)1 / time * input[i]);
        }
    }

    /**
     * Compute error of a vector of floats relative to another vector of floats
     *
     * @param originalVector The original vector used as control
     * @param  compareVector The vector we would like to compute the error of
     * @return relativeError The error of compareVector relative to originalVector
     */
    private double computeRelativeError(float[] originalVector, float[] compareVector)
    {
        // Compute euclidean norm of original vector and difference vector
        double originalNorm = 0;
        double differenceNorm = 0;
        float difference;
        for (int i = 0; i < originalVector.length; ++i)
        {
            originalNorm += pow(originalVector[i], 2);

            difference = abs(originalVector[i] - compareVector[i]);
            differenceNorm += pow(difference,2 );
        }
        originalNorm = sqrt(originalNorm);
        differenceNorm = sqrt(differenceNorm);

        // Compute relative error: ratio between euclidean norms
        double relativeError = differenceNorm / originalNorm;
        return relativeError;
    }

    // Used to load the 'native-lib' and 'OpenCL' libraries on application startup.
    static {
        System.loadLibrary("native-lib");
        // Load OpenCL library from absolute filepath currently based on Adreno powered GPU.
        System.load("/system/vendor/lib/libOpenCL.so");
    }

}
