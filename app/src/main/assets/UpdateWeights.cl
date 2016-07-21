kernel void fillZero(__global float* w)
{
    w[get_global_id(0)] = 0;
}
kernel void UpdateWeights(__global float* w, __constant float *input, __private int t)
{
    int globalIndex = get_global_id(0);
    w[globalIndex] = ( (float)(t-1)/t * w[globalIndex]) + ((float)1/t * input[globalIndex]);
}