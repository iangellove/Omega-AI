#define BLOCK 1024 

#include <cuda_fp16.h>

__device__ __half __hmul(const __half a, const __half b){
  return __float2half(__half2float(a)*__half2float(b));
}

extern "C"
__global__ void flash_attention_2_forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    const int q_start_offset,
    float* L,
    float* O
) {
    const float INF = 9999999999.9f; 
    int tx = threadIdx.x;
    int txd = tx * d;

    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index
    int bz = blockIdx.z; // Tr index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for L

    // Define SRAM for Q,K,V,S
    extern __shared__ float sramb[];
    __half *sram = (__half *)sramb;

    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    __half* Qi = sram;
    __half* KVj = &sram[tile_size];

    int i = bz;
    if (i >= q_start_offset && i < Tr)
    {       
        if (i * Br + tx >= N)
            return;  // break if we are done with the sequence

        // Load Qi from HBM to SRAM, l and m to registers

        for (int x = 0; x < d; x++) {
            Qi[txd + x] = __float2half(Q[qkv_offset + (tile_size * i) + txd + x]);
        }
        float row_m_prev = -INF;
        float row_l_prev = 0;
        float lS[256];

        // Causal mask: j <= i
        for (int j = 0; j <= i; ++j) {
            __syncthreads();
            // Load Kj, Vj from HBM to SRAM

            for (int x = 0; x < d; x++) {
                KVj[txd + x] = __float2half(K[qkv_offset + (tile_size * j) + txd + x]);
            }
            
            __syncthreads();

            int yMax = min(min(Bc, N - j * Bc), i * Br - j * Bc + tx + 1);         

            // S_i^j = softmax_scale * QiKj^T
            // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
            float row_m = -INF;

            for (int y = 0; y < yMax; y++) {
                //if (j * Bc + y >= N)
                //    break;  // break if we are done with the sequence
                //if (i * Br + tx < j * Bc + y)
                //    break;
                float sum = 0;

                    for (int x = 0; x < d; x++)
                        sum += __half2float(__hmul(Qi[txd + x], KVj[(y * d) + x]));
                 
                sum *= softmax_scale;
                lS[y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // m_i^j = max(m_i^j-1, row_max(S_i^j))
            float new_row_m = max(row_m_prev, row_m);

            // P_i^j = exp(S_i^j - m_i^j)
            // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
            float row_l = 0;
            for (int y = 0; y < yMax; y++) {
                //if (j * Bc + y >= N)
                //    break;  // break if we are done with the sequence
                //if (i * Br + tx < j * Bc + y)
                //    break;

                float r = __expf(lS[y] - new_row_m);
                lS[y] = r;
                row_l += r;
            }

            __syncthreads();
            for (int x = 0; x < d; x++) {
                KVj[txd + x] = __float2half(V[qkv_offset + (tile_size * j) + txd + x]);
            }
            __syncthreads();

            // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
            float row_m_exp = __expf(row_m_prev - new_row_m);
            float new_row_l = (row_m_exp * row_l_prev) + row_l;

            // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < yMax; y++) {
                    //if (j * Bc + y >= N)
                    //    break;  // break if we are done with the sequence
                    //if (i * Br + tx < j * Bc + y)
                    //    break;
                    pv += lS[y] * __half2float(KVj[(y * d) + x]);
                }
                O[qkv_offset + (tile_size * i) + txd + x] = \
                    row_m_exp * O[qkv_offset + (tile_size * i) + txd + x] + pv;
            }

            // Update m and l
            row_m_prev = new_row_m;
            row_l_prev = new_row_l;
        }

        // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
        for (int x = 0; x < d; x++){
            O[qkv_offset + (tile_size * i) + txd + x] /= row_l_prev;
        }    
        // L_i = m_i^{Tc} + log(l_i^{Tc})
        L[lm_offset + (Br * i) + tx] = row_m_prev + __logf(row_l_prev);
    }
}

extern "C"
__global__ void flash_attention_2_backward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* dO,
    const float* L,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* dQ,
    float* dK,
    float* dV,
    float* Stmp
) {
    const float INF = 9999999999.9f; 
    int tx = threadIdx.x;
    int txd = tx * d;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index
    int bz = blockIdx.z; // Tc index;

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for L

    // Define SRAM for Q,K,V,S
    extern __shared__ float sramb[];
    __half* sram = (__half*)sramb;

    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    __half* Kj = sram;
    __half* Vj = &sram[col_tile_size];

    __half* Qi = &sram[col_tile_size * 2];
    __half* dOi = &sram[col_tile_size * 2 + row_tile_size];

    // We also use S for P. Likewise, we use dS for dP.
    // We can reuse the same memory because we don't need S and P at the same time.
    // We also don't need dS and dP at the same time.
    //__half* S = &sram[col_tile_size * 2 + row_tile_size * 2];

    int stmp_offset = (bx * gridDim.y * Br * Br) + (by * Br * Br);  // gridDim.y = nh
    float* S = &Stmp[stmp_offset];

     int j = bz;
     if (j < Tc) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[txd + x] = __float2half(K[qkv_offset + (col_tile_size * j) + txd + x]);
            Vj[txd + x] = __float2half(V[qkv_offset + (col_tile_size * j) + txd + x]);
        }

        for (int i = j; i < Tr; i++)  {
            __syncthreads();
            // Load Qi, Oi, dOi, dQi, li, mi to SRAM
            // Also load l, m to registers
            float Di = 0;

                for (int x = 0; x < d; x++) {
                    Qi[txd + x] = __float2half(Q[qkv_offset + (row_tile_size * i) + txd + x]);
                    float dO_v = dO[qkv_offset + (row_tile_size * i) + txd + x];
                    dOi[txd + x] = __float2half(dO_v);
                    Di += dO_v * O[qkv_offset + (row_tile_size * i) + txd + x];
                }
            

            float l_curr = L[lm_offset + (Br * i) + tx];

            // Sij = softmax_scale * QiKj^T
            // Sij[tx][y] = softmax_scale * Sum_{y = 0}^{Bc-1} Qi[tx][x] * Kj[y][x]

            // Pij = diag(li)^-1 * exp(Sij - mi)
            // Pij[tx][y] = (1 / li[tx]) * exp(Sij[tx][y] - mi[tx])

            for (int y = 0; y < Bc; y++) {
                float sum = 0;

                    for (int x = 0; x < d; x++) {
                        sum += __half2float(__hmul(Qi[txd + x], Kj[(y * d) + x]));
                    }
                
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = __expf(sum - l_curr);
            }

            __syncthreads();
            // dVj <- dVj + Pij^T * dOi
            // dVj[tx][x] = dVj[tx][x] + Sum_{y = 0}^{Br-1} Pij[y][tx] * dOi[tx][x]
            for (int x = 0; x < d; x++) {
                float sum = 0;
                float dOi_x = __half2float(dOi[txd + x]);              

                    for (int y = 0; y < Br; y++) {
                        sum += S[(Bc * y) + tx] * dOi_x;
                    }
                

                atomicAdd(&dV[qkv_offset + (row_tile_size * j) + txd + x], sum);
            }

            // dPij <- dOi * Vj^T
            // dPij[tx][y] = Sum_{x = 0}^{d-1} dOi[tx][x] * Vj[y][x]

            // dSij <- Pij * (dPij - Di)
            // dSij[tx][y] = Pij[tx][y] * (dPij[tx][y] - Di[tx])
            for (int y = 0; y < Bc; y++) {
                float sum = 0;


                    for (int x = 0; x < d; x++) {
                        sum += __half2float(__hmul(dOi[txd + x], Vj[(y * d) + x]));
                    }
                

                S[(Bc * tx) + y] = S[(Bc * tx) + y] * (sum - Di);
            }

            // dQi <- dQi + softmax_scale * dSijKj
            // dQ[tx][x] = dQ[tx][x] + softmax_scale * Sum_{y = 0}^{Bc-1} dSij[tx][y] * Kj[y][x]
            for (int x = 0; x < d; x++) {
                float sum = 0;

                for (int y = 0; y < Bc; y++) {
                    sum += S[(Bc * tx) + y] * __half2float(Kj[(y * d) + x]);
                }
                sum *= softmax_scale;
                atomicAdd(&dQ[qkv_offset + (row_tile_size * i) + txd + x], sum);
            }
            __syncthreads();
            // dKj <- dKj + softmax_scale * dSij^TQi
            // dKj[tx][x] = dKj[tx][x] + softmax_scale * Sum_{y = 0}^{Br-1} dSij[y][tx] * Qi[y][x]
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < Br; y++) {
                    sum += S[(Bc * y) + tx] * __half2float(Qi[(y * d) + x]);
                }

                sum *= softmax_scale;
                atomicAdd(&dK[qkv_offset + (row_tile_size * j) + txd + x], sum);
            }
        }
    }
}