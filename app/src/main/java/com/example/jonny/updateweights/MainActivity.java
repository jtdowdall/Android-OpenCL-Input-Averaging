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

public class MainActivity extends AppCompatActivity {

    /** Maximum size of W and input vectors */
    public int mSizeW;

    /** Kernel filename for OpenCL to load */
    public String mKernelName = "UpdateWeights.cl";

    /** Input array used for updating weights */
    public float mInput[];

    /** Array storing input averages for CPU computation */
    public float mCpuW[];


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
                        long start, end;
                        for (int time = 1; time <= finalTime; ++time ) {
                            // Create mock input using random array of floats
                            randomizeFloats(mInput);

                            // Update CPU weights and keep track of runtime
                            start = System.nanoTime();
                            cpuUpdateWeights(time);
                            end = System.nanoTime();
                            cpuTime += (double)(end - start) / 1000000;

                            // Update GPU weights and keep track of runtime
                            start = System.nanoTime();
                            updateWeights(mInput, time);
                            end = System.nanoTime();
                            gpuTime += (double)(end - start) / 1000000;
                        }

                        //String result = getResults();

                        String result = "CPU: " + Double.toString(cpuTime) + " ms\n" +
                                "GPU: " + Double.toString(gpuTime) + " ms";
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
    public native String getResults();

    /*
	 * loads the kernel into the app_execdir
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
    private void cpuUpdateWeights(int t)
    {
        for (int i = 0; i < mSizeW; ++i) {
            mCpuW[i] = ((float)(t - 1) / t * mCpuW[i]) + ((float)1 / t * mInput[i]);
        }
    }
    // Used to load the 'native-lib' and 'OpenCL' libraries on application startup.
    static {
        System.loadLibrary("native-lib");
        // Load OpenCL library from absolute filepath currently based on Adreno powered GPU.
        System.load("/system/vendor/lib/libOpenCL.so");
    }

}
