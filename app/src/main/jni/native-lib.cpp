/**
 * native-lib.cpp
 * @author Jonathan Dowdall
 * @since 06-15-2016
 */

#include <jni.h>
#include <CL/cl.h>
#include <string>
#include <fstream>
#include <iostream>
#include <android/log.h>
#include <vector>
#include <random>
#include <ctime>
#include <cstdlib>

/* Container for all OpenCL-specific objects used.
 *
 * The container consists of the following parts:
 *   - Regular OpenCL objects, used in almost each
 *     OpenCL application.
 *   - Specific OpenCL objects - buffers, used in this
 *     particular sample.
 *
 * For convenience, collect all objects in one structure.
 * Avoid global variables and make easier the process of passing
 * all arguments in functions.
 *
 * As defined by Kronos specification:
 * https://www.khronos.org/registry/cl/specs/opencl-1.0.29.pdf
 */
struct OpenCLObjects
{
    /** The platform consists of one or more OpenCL devices */
    cl_platform_id platform;

    /** A device is a collection of compute units. A command-queue is used to queue
     * commands to a device. Examples of commands include executing kernels,
     * or reading and writing memory objects. OpenCL devices typically correspond to a GPU,
     * a multi-core CPU, and other processors such as DSPs and the Cell/B.E. processor. */
    cl_device_id device;

    /** The environment within which the kernels execute and
     * the domain in which synchronization and memory management is defined.
     * The context includes a set of devices, the memory accessible to those devices,
     * the corresponding memory properties and one or more command-queues used to
     * schedule execution of a kernel(s) or operations on memory objects.
     */
    cl_context context;

    /** A data structure used to coordinate execution of the kernels on the devices.
     * The host places commands into the command-queue
     * which are then scheduled onto the devices within the context.
     */
    cl_command_queue queue;

    /** An object that encapsulates the following:
     *      - A reference to an associated context.
     *      - A program source or binary.
     *      - The latest successfully built program executable,
     *        the list of devices for which the program executable
     *        is built, the build options used and a build log.
     *      - The number of kernel objects currently attached.
     */
    cl_program program;

    /** Kernel used to update elements in array W when provided new input vector. */
    cl_kernel updateWeights;
};

/** The GPU properties provided by OpenCL APU queries.
 *
 * These properties are important to properly manage problem dimensions
 * and memory management.
 */
struct GpuProperties{
    /** The name of the GPU device */
    char name[128];

    /** How many compute units (GPU cores) are on the device */
    cl_int computeUnits;

    /** Maximum global memory size */
    cl_ulong globalMem;

    /** Maximum local memory size */
    cl_ulong localMem;

    /** Maximum size of memory allocation for buffers */
    cl_ulong maxAllocSize;

    /** True if GPU shares memory with host (integrated graphics)
     *  False if GPU has dedicated memory (discrete graphics)
     *      -Information must be transferred between host and GPU
     */
    cl_bool unifiedMem;
};

struct W{
    /** Memory allocated for array w */
    cl_mem buffer;

    /** Pointer to mapped to allocated memory for array W */
    float* pointer;

    /** Size of array */
    int size;
};


/** Global cl variable to store context among functions */
OpenCLObjects cl;

/** Global GPU properties to share between initialization and kernel designs */
GpuProperties gpu;

/** Global array that maintains input average for GPU computation. */
W wGpu;

/** Global array that maintains input average for CPU computation. */
float *wCpu;

/** Global array of input vector. */
W inputVector;

/** Global variables to keep track of elapsed time for cpu/gpu functions */
double cpuTime = 0;
double gpuTime = 0;

/** Global variable keeping track of time iterations for updating weights */
unsigned int t = 0;

/********************************** Helper Functions ********************************************/
// Commonly-defined shortcuts for LogCat output from native C applications.
#define  LOG_TAG    "AndroidBasic"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/* This function helps to create informative messages in
 * case when OpenCL errors occur. The function returns a string
 * representation for an OpenCL error code.
 * For example, "CL_DEVICE_NOT_FOUND" instead of "-1".
 */
const char* opencl_error_to_str (cl_int error)
{
#define CASE_CL_CONSTANT(NAME) case NAME: return #NAME;

// Suppose that no combinations are possible.
switch(error)
{
CASE_CL_CONSTANT(CL_SUCCESS)
CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
CASE_CL_CONSTANT(CL_MAP_FAILURE)
CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
CASE_CL_CONSTANT(CL_INVALID_VALUE)
CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
CASE_CL_CONSTANT(CL_INVALID_DEVICE)
CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
CASE_CL_CONSTANT(CL_INVALID_BINARY)
CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
CASE_CL_CONSTANT(CL_INVALID_KERNEL)
CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
CASE_CL_CONSTANT(CL_INVALID_EVENT)
CASE_CL_CONSTANT(CL_INVALID_OPERATION)
CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
CASE_CL_CONSTANT(CL_INVALID_PROPERTY)

default:
return "UNKNOWN ERROR CODE";
}

#undef CASE_CL_CONSTANT
}

/* The following macro is used after each OpenCL call
 * to check if OpenCL error occurs. In the case when ERR != CL_SUCCESS
 * the macro forms an error message with OpenCL error code mnemonic,
 * puts it to LogCat, and returns from a caller function.
 *
 * The approach helps to implement consistent error handling tactics
 * because it is important to catch OpenCL errors as soon as
 * possible to avoid missing the origin of the problem.
 *
 * You may chose a different way to do that. The macro is
 * simple and context-specific as it assumes you use it in a function
 * that doesn't have a return value, so it just returns in the end.
 */
#define SAMPLE_CHECK_ERRORS(ERR)                                                      \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGE                                                                          \
        (                                                                             \
            "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
            opencl_error_to_str(ERR), __FILE__, __LINE__                              \
        );                                                                            \
                                                                                      \
        return 0;                                                                     \
    }

/*
 * Load the program out of the file in to a string for opencl compiling.
 */
inline std::string loadProgram(std::string input)
{
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        LOGE("Cannot open input file\n");
        exit(1);
    }
    return std::string( std::istreambuf_iterator<char>(stream),
                        (std::istreambuf_iterator<char>()));
}

/********************************** /Helper Functions ********************************************/

enum NativeType
{
    JBoolean,
    JByte,
    JChar,
    JShort,
    JInt,
    JLong,
    Jfloat,
    JDouble,
};

extern "C" JNIEXPORT jint JNICALL
Java_com_example_jonny_updateweights_MainActivity_initOpenCl(JNIEnv *env, jobject instance, jstring kernelName) {
    // The following variable stores return codes for all OpenCL calls.
    // In the code it is used with the SAMPLE_CHECK_ERRORS macro defined
    // before this function.
    cl_int err = CL_SUCCESS;

    /* -----------------------------------------------------------------------
     * Step 1: Query and choose OpenCL platform.
     */
    clGetPlatformIDs(1, &cl.platform, NULL);
    char platform_name[100];
    err = clGetPlatformInfo(
            cl.platform,
            CL_PLATFORM_NAME,
            100,
            platform_name,
            0
    );
    LOGD("Platform: %s", platform_name);

    /* -----------------------------------------------------------------------
     * Step 2: Create context with a device of the specified type.
     * Required device type is passed as function argument required_device_type.
     * Use this function to create context for any CPU or GPU OpenCL device.
     */
    cl_context_properties context_props[] = {
            CL_CONTEXT_PLATFORM,
            cl_context_properties(cl.platform),
            0
    };
    cl.context = clCreateContextFromType
            (
                    context_props,
                    CL_DEVICE_TYPE_GPU, //Searching sepcifically for GPU
                    0,
                    0,
                    &err
            );
    if (err == CL_DEVICE_NOT_AVAILABLE || err == CL_DEVICE_NOT_FOUND) return 0;
    SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * Step 3: Query for OpenCL device that was used for context creation.
     */
    err = clGetContextInfo
            (
                    cl.context,
                    CL_CONTEXT_DEVICES,
                    sizeof(cl.device),
                    &cl.device,
                    0
            );

    err = clGetDeviceInfo
            (
                    cl.device,
                    CL_DEVICE_NAME,
                    sizeof(gpu.name),
                    gpu.name,
                    NULL
            );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("CL_DEVICE_NAME: %s", gpu.name );

    err = clGetDeviceInfo
            (
                    cl.device,
                    CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(gpu.computeUnits),
                    &gpu.computeUnits,
                    NULL
            );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("Total Cores: %d", gpu.computeUnits );

    err = clGetDeviceInfo
            (
                    cl.device,
                    CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(gpu.globalMem),
                    &gpu.globalMem,
                    NULL
            );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("Global Memory Size (bytes): %lu", (unsigned long) gpu.globalMem );

    err = clGetDeviceInfo
            (
                    cl.device,
                    CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(gpu.localMem),
                    &gpu.localMem,
                    NULL
            );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("Local Memory Size (bytes): %lu", (unsigned long) gpu.localMem );

    err = clGetDeviceInfo
            (
                    cl.device,
                    CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(gpu.maxAllocSize),
                    &gpu.maxAllocSize,
                    NULL
            );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("Maximum memory allocation (bytes): %lu", (unsigned long) gpu.maxAllocSize );

    err = clGetDeviceInfo
            (
                    cl.device,
                    CL_DEVICE_HOST_UNIFIED_MEMORY,
                    sizeof(gpu.unifiedMem),
                    &gpu.unifiedMem,
                    NULL
            );
    SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * Step 4: Create OpenCL program from its source code.
     * The file name is passed by java.
     * Convert the jstring to const char* and append the needed directory path.
     */
    const char* fileName = env->GetStringUTFChars(kernelName, 0);
    std::string fileDir;
    fileDir.append("/data/data/com.example.jonny.updateweights/app_execdir/");
    fileDir.append(fileName);
    std::string kernelSource = loadProgram(fileDir);
    //std::string to const char* needed for the clCreateProgramWithSource function
    const char* kernelSourceChar = kernelSource.c_str();

    cl.program = clCreateProgramWithSource
            (
                    cl.context,
                    1,
                    &kernelSourceChar,
                    0,
                    &err
            );

    SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * Step 5: Build the program.
     * During creation a program is not built. Call the build function explicitly.
     * This example utilizes the create-build sequence, still other options are applicable,
     * for example, when a program consists of several parts, some of which are libraries.
     * Consider using clCompileProgram and clLinkProgram as alternatives.
     * Also consider looking into a dedicated chapter in the OpenCL specification
     * for more information on applicable alternatives and options.
     */
    err = clBuildProgram(cl.program, 0, 0, NULL, 0, 0);
    SAMPLE_CHECK_ERRORS(err);
    if(err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_length = 0;
        err = clGetProgramBuildInfo
                (
                        cl.program,
                        cl.device,
                        CL_PROGRAM_BUILD_LOG,
                        0,
                        0,
                        &log_length
                );
        SAMPLE_CHECK_ERRORS(err);

        std::vector<char> log(log_length);

        err = clGetProgramBuildInfo
                (
                        cl.program,
                        cl.device,
                        CL_PROGRAM_BUILD_LOG,
                        log_length,
                        &log[0],
                        0
                );
        SAMPLE_CHECK_ERRORS(err);

        LOGE
        (
                "Error happened during the build of OpenCL program.\nBuild log:%s",
                &log[0]
        );
        return 0;
    }

    /* -----------------------------------------------------------------------
     * Step 6: Extract kernel from the built program.
     * An OpenCL program consists of kernels. Each kernel can be called (enqueued) from
     * the host part of an application.
     * First create a kernel to call it from the existing program.
     * Creating a kernel via clCreateKernel is similar to obtaining an entry point of a specific function
     * in an OpenCL program.
     */
    cl.updateWeights = clCreateKernel
            (
                    cl.program,
                    "UpdateWeights",
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * Step 7: Create command queue.
     * OpenCL kernels are enqueued for execution to a particular device through
     * special objects called command queues. Command queue provides ordering
     * of calls and other OpenCL commands.
     * This sample uses a simple in-order OpenCL command queue that doesn't
     * enable execution of two kernels in parallel on a target device.
     */
    cl.queue = clCreateCommandQueue
            (
                    cl.context,
                    cl.device,
                    CL_QUEUE_PROFILING_ENABLE,
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    if (gpu.unifiedMem == 1) return 2;

    return 1;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_jonny_updateweights_MainActivity_initW(JNIEnv *env, jobject instance) {

    cl_int err;

    wGpu.size = gpu.maxAllocSize / 2 / sizeof(double);

    // If unified memory, allocate memory on host. Otherwise allocate on GPU memory.
    if (gpu.unifiedMem) {
        // Since global memory is shared with host,
        // account for memory limit on host as well
        wGpu.size /= 2;

        // Create OpenCL memory buffer for vector w in host memory
        wGpu.buffer = clCreateBuffer
                (
                        cl.context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        wGpu.size * sizeof(float),
                        NULL,
                        &err
                );
        SAMPLE_CHECK_ERRORS(err);

        // Create OpenCL memory buffer for input vector in host memory
        inputVector.size = wGpu.size;

        SAMPLE_CHECK_ERRORS(err);
    }

    else
    {
        // Create OpenCL memory buffer for input vector in GPU memory
        wGpu.buffer = clCreateBuffer
                (
                        cl.context,
                        CL_MEM_WRITE_ONLY,
                        wGpu.size * sizeof(float),
                        NULL,
                        &err
                );
        SAMPLE_CHECK_ERRORS(err);

        // Create OpenCL memory buffer for input vector in GPU memory
        inputVector.size = wGpu.size;
        inputVector.buffer = clCreateBuffer
                (
                        cl.context,
                        CL_MEM_READ_ONLY,
                        inputVector.size * sizeof(float),
                        NULL,
                        &err
                );
    }

    // Fill GPU array with zeros in parallel
    cl_kernel fillZero = clCreateKernel(cl.program, "fillZero", &err);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalDimensions[3] = {wGpu.size, 1, 1};

    err = clSetKernelArg
            (
                    fillZero, // kernel
                    0, // arg_index
                    sizeof(wGpu.buffer), // arg_size
                    &wGpu.buffer // *arg_value
            );
    SAMPLE_CHECK_ERRORS(err);

    err = clEnqueueNDRangeKernel
            (
                    cl.queue, // command_queue
                    fillZero, // kernel
                    3, // work_dim
                    NULL, // *global_work_offset
                    globalDimensions, // *globa_work_size
                    NULL, // *local_work_size
                    0, // num_events_in_wait_list
                    NULL, // *event_wait_list
                    NULL // *event
            );
    SAMPLE_CHECK_ERRORS(err);

    // Create CPU array
    wCpu = new float[wGpu.size];
    std::fill(wCpu, wCpu + wGpu.size, 0);
    LOGD("%f",wCpu[0]);

    return (int) wGpu.size;

}

extern "C" JNIEXPORT int
Java_com_example_jonny_updateweights_MainActivity_updateWeights(JNIEnv *env, jobject instance,
                                                                   jfloatArray input, jint t) {
    cl_int err;

    // Create buffer for input vector
    inputVector.pointer = (float *) env->GetFloatArrayElements(input, 0);
    inputVector.buffer = clCreateBuffer
            (
                    cl.context,
                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    inputVector.size * sizeof(float),
                    inputVector.pointer,
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    // Set kernel arguments
    err = clSetKernelArg
            (
                    cl.updateWeights,
                    0,
                    sizeof(wGpu.buffer),
                    &wGpu.buffer
            );
    err |= clSetKernelArg
            (
                    cl.updateWeights,
                    1,
                    sizeof(inputVector.buffer),
                    &inputVector.buffer
            );
    err |= clSetKernelArg
            (
                    cl.updateWeights,
                    2,
                    sizeof(int),
                    &t
            );
    SAMPLE_CHECK_ERRORS(err);

    // Run kernel
    size_t globalDimensions[3] = {wGpu.size, 1, 1};
    err = clEnqueueNDRangeKernel
            (
                    cl.queue, // command_queue
                    cl.updateWeights, // kernel
                    3, // work_dim
                    NULL, // *global_work_offset
                    globalDimensions, // *globa_work_size
                    NULL, // *local_work_size
                    0, // num_events_in_wait_list
                    NULL, // *event_wait_list
                    NULL // *event
            );
    SAMPLE_CHECK_ERRORS(err);

    return 1;

}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_jonny_updateweights_MainActivity_getGpuW(JNIEnv *env, jobject instance) {

    cl_int err;
    wGpu.pointer = (float*)clEnqueueMapBuffer
            (
                    cl.queue, // command_queue
                    wGpu.buffer, // buffer
                    true, // blocking_map
                    CL_MAP_READ, // maps_flags
                    0, // offset
                    wGpu.size * sizeof(float), // cb
                    0, // num_events_in_wait_list
                    NULL, // *event_wait_list
                    NULL, // *event
                    &err // *errcode_ret
            );
    SAMPLE_CHECK_ERRORS(err);

    jfloatArray result = env->NewFloatArray(wGpu.size);
    env->SetFloatArrayRegion(result, 0, wGpu.size, wGpu.pointer);

    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_jonny_updateweights_MainActivity_getResults(JNIEnv *env, jobject instance) {

    cl_int err;
    wGpu.pointer = (float*)clEnqueueMapBuffer
            (
                    cl.queue, // command_queue
                    wGpu.buffer, // buffer
                    true, // blocking_map
                    CL_MAP_READ, // maps_flags
                    0, // offset
                    wGpu.size * sizeof(float), // cb
                    0, // num_events_in_wait_list
                    NULL, // *event_wait_list
                    NULL, // *event
                    &err // *errcode_ret
            );
    SAMPLE_CHECK_ERRORS(err);

    double cpuNorm = 0.0;
    double differenceNorm = 0.0;

    for (int i = 0; i < 10; ++i){
        LOGD("CPU: %f GPU: %f", wCpu[i], wGpu.pointer[i]);
    }

    for (int i = 0; i < wGpu.size; ++i){
        differenceNorm += pow(abs(wCpu[i] - wGpu.pointer[i]),2);
        cpuNorm += pow(wCpu[i], 2);
    }
    cpuNorm = sqrt(cpuNorm);
    differenceNorm = sqrt(differenceNorm);

    double relativeError = differenceNorm / cpuNorm;

    std::string result;
    result += "Results:\n";
    result += std::to_string(wGpu.size) + " elements were updated " +
            std::to_string(t-1) + " time(s) to maintain input averages.\n";
    result += "\nCPU: " + std::to_string(cpuTime) + " ms";
    result += "\nGPU: " + std::to_string(gpuTime) + " ms";
    result += "\nRuntime reduction: " + std::to_string((double)(1 - gpuTime/cpuTime) * 100) + "%\n";
    result += "\nGPU relative error to CPU: " +  std::to_string(relativeError*100) + "%";
    result += "\nwCpu[0]: " + std::to_string(wCpu[0]);
    result += "\nwGpu[0]: " + std::to_string(wGpu.pointer[0]);
    result += "\nwCpu[1]: " + std::to_string(wCpu[1]);
    result += "\nwGpu[1]: " + std::to_string(wGpu.pointer[1]);

    return env->NewStringUTF(result.c_str());
}