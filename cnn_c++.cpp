#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>  
#include <memory>
#include <cstdio>

#define STRIDE 1
#define IN_SCALE 0.025556690990924835
#define IN_ZERO 17
#define W_SCALE 0.004168602637946606
#define W_ZERO 0
#define OUT_SCALE 0.102776609361171722
#define OUT_ZERO 0

#define O_CH1 32
#define O_CH2 64
#define O_CH3 128
#define O_CH4 128
#define O_CH5 256
#define O_CH6 256
#define INPUT_CHANNELS 3
#define Rx 32
#define Cx 32
#define KRx 3
#define KCx 3

std::vector<float> load_params_new(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<float> params;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float param;
        while (iss >> param) {
            params.push_back(param);
        }
    }

    if (params.empty()) {
        std::cerr << "Error: No data found in file " << file_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    //std::cout << "Loaded " << params.size() << " parameters from " << file_path << std::endl;
    return params;
}

// �ı亯���������ض����ͻ
void load_and_reshape_weights_new(const std::string& file_path, signed char* weights, int size) {
    std::vector<float> weight_vec = load_params_new(file_path);
    if (weight_vec.size() < size) {
        std::cerr << "Error: Not enough data in weight file." << std::endl;
        std::cerr << "Expected size: " << size << ", but got: " << weight_vec.size() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // �ҳ�����Ȩ�ؾ���ֵ
    float max_weight = 0.0f;
    for (float w : weight_vec) {
        if (std::abs(w) > max_weight) {
            max_weight = std::abs(w);
        }
    }

    // �������������Խ�Ȩ��ӳ�䵽[-127, 127]��Χ��
    float scale_factor = 127.0f / max_weight;
    //std::cout << "Scale factor: " << scale_factor << std::endl;

    for (int i = 0; i < size; ++i) {
        weights[i] = static_cast<signed char>(std::round(weight_vec[i] * scale_factor));
        //if (i < 10) {
        //    std::cout << (int)weights[i] << " ";
        //}
    }
    std::cout << std::endl;
}   //7/17�������ҵ�bias��weight������ȷ׼ȷ�ļ��ؽ�ģ�����ˡ�
    //for (int i = 0; i < size; ++i) {
    //    weights[i] = static_cast<signed char>(weight_vec[i]);
    //}
    //std::cout << "First 10 weights before conversion: ";
    //for (int i = 0; i < std::min(10, (int)weight_vec.size()); ++i) {
    //    std::cout << weight_vec[i] << " ";
    //}
    //std::cout << std::endl;

   // for (int i = 0; i < size; ++i) {
        // Ӧ�����ź�ƫ��
        //float scaled_weight = weight_vec[i] * W_SCALE + W_ZERO;
        // ȷ��ֵ��signed char�ķ�Χ��
        //scaled_weight = std::min(std::max(scaled_weight, -128.0f), 127.0f);
        //weights[i] = static_cast<signed char>(std::round(scaled_weight));  // Ϊ�˰�ȫ���������std::round
        //if (i < 10) {  // ͬ����ӡת�����ǰ10��Ȩ��ֵ
        //    std::cout << (int)weights[i] << " ";
        //}
    //}
    //std::cout << std::endl;


void load_and_reshape_bias(const std::string& file_path, float* bias, int size) {
    std::vector<float> bias_vec = load_params_new(file_path);
    if (bias_vec.size() < size) {
        std::cerr << "Error: Not enough data in bias file." << std::endl;
        std::cerr << "Expected size: " << size << ", but got: " << bias_vec.size() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; ++i) {
        bias[i] = bias_vec[i];
    }
}

// Convolutional layer weights and biases
signed char weight1[O_CH1][INPUT_CHANNELS][KRx][KCx];
float bias1[O_CH1];
float mean1[O_CH1];
float variance1[O_CH1];
float gamma1[O_CH1];
float beta1[O_CH1];

signed char weight2[O_CH2][O_CH1][KRx][KCx];
float bias2[O_CH2];
float mean2[O_CH2];
float variance2[O_CH2];
float gamma2[O_CH2];
float beta2[O_CH2];

signed char weight3[O_CH3][O_CH2][KRx][KCx];
float bias3[O_CH3];
float mean3[O_CH3];
float variance3[O_CH3];
float gamma3[O_CH3];
float beta3[O_CH3];

signed char weight4[O_CH4][O_CH3][KRx][KCx];
float bias4[O_CH4];
float mean4[O_CH4];
float variance4[O_CH4];
float gamma4[O_CH4];
float beta4[O_CH4];

signed char weight5[O_CH5][O_CH4][KRx][KCx];
float bias5[O_CH5];
float mean5[O_CH5];
float variance5[O_CH5];
float gamma5[O_CH5];
float beta5[O_CH5];

signed char weight6[O_CH6][O_CH5][KRx][KCx];
float bias6[O_CH6];
float mean6[O_CH6];
float variance6[O_CH6];
float gamma6[O_CH6];
float beta6[O_CH6];

// Fully connected layer weights and biases
float weights_fc1[512 * (O_CH6 * Rx / 16 * Cx / 16)];
float bias_fc1[512];
float weights_fc2[256 * 512];
float bias_fc2[256];
float weights_fc3[10 * 256];
float bias_fc3[10];

template<int O_CH, int I_CH, int KR, int KC>
void conv_layer(uint8_t* in_img, uint8_t* out_img, int rows, int cols, float* bias, int8_t(&weight)[O_CH][I_CH][KR][KC], const std::string& layer_name) {
    std::cout << "Starting conv_layer for O_CH: " << O_CH << ", I_CH: " << I_CH << std::endl;

    // Padding the input image
    int padded_rows = rows + 2;
    int padded_cols = cols + 2;
    std::vector<uint8_t> in_arr(I_CH * padded_rows * padded_cols, 0);
    std::vector<uint8_t> out_arr(O_CH * rows * cols, 0);  // Initialize output array with zeros

    // Copy input image into padded array
    for (int i_ch = 0; i_ch < I_CH; ++i_ch) {
        for (int r = 0; r < rows; ++r) {
            memcpy(&in_arr[i_ch * padded_rows * padded_cols + (r + 1) * padded_cols + 1],
                &in_img[i_ch * rows * cols + r * cols], cols);
        }
    }

    // Perform the convolution
    for (int o_ch = 0; o_ch < O_CH; ++o_ch) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float sum = 0.0;
                for (int i_ch = 0; i_ch < I_CH; ++i_ch) {
                    for (int kr = 0; kr < KR; ++kr) {
                        for (int kc = 0; kc < KC; ++kc) {
                            int in_val = in_arr[i_ch * padded_rows * padded_cols + (r + kr) * padded_cols + (c + kc)];
                            int w_val = weight[o_ch][i_ch][kr][kc];
                            sum += ((in_val - IN_ZERO) * IN_SCALE) * ((w_val - W_ZERO) * W_SCALE);
                        }
                    }
                }
                sum += bias[o_ch];

                // Print intermediate sum before activation
                if (r == 0 && c == 0 && o_ch == 0) {
                    std::cout << "sum (before activation) for output channel " << o_ch << ": " << sum << std::endl;
                }

                // Apply Leaky ReLU
                sum = sum > 0 ? sum : 0.01 * sum;
                int out_val = static_cast<int>(round(sum / OUT_SCALE + OUT_ZERO));
                out_arr[o_ch * rows * cols + r * cols + c] = static_cast<uint8_t>(std::max(0, std::min(255, out_val)));

                // Print intermediate sum after activation
                if (r == 0 && c == 0 && o_ch == 0) {
                    std::cout << "sum (after activation) for output channel " << o_ch << ": " << sum << " out_val: " << out_val << std::endl;
                }
            }
        }
    }

    memcpy(out_img, out_arr.data(), sizeof(uint8_t) * O_CH * rows * cols);
    std::cout << "Completed conv_layer for O_CH: " << O_CH << std::endl;
}



template<int O_CH>
void bn_layer(uint8_t * in_img, uint8_t * out_img, int rows, int cols, float* mean, float* variance, float* gamma, float* beta, const std::string & layer_name) {
    std::cout << "Starting bn_layer for O_CH: " << O_CH << std::endl;

    std::vector<uint8_t> in_arr(O_CH * rows * cols);
    std::vector<uint8_t> out_arr(O_CH * rows * cols);

    memcpy(in_arr.data(), in_img, sizeof(uint8_t) * O_CH * rows * cols);

    for (int o_ch = 0; o_ch < O_CH; ++o_ch) {
       // std::cout << "Processing channel " << o_ch << std::endl;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float normalized = (in_arr[o_ch * rows * cols + r * cols + c] - mean[o_ch]) / sqrt(variance[o_ch] + 1e-5);
                float scaled = gamma[o_ch] * normalized + beta[o_ch];
                scaled = std::max(0.0f, scaled); // Apply ReLU
                out_arr[o_ch * rows * cols + r * cols + c] = static_cast<uint8_t>(round(scaled));
            }
        }
    }

    memcpy(out_img, out_arr.data(), sizeof(uint8_t) * O_CH * rows * cols);
    std::ofstream outfile(layer_name + "_output.txt");
    for (int i = 0; i < O_CH * rows * cols; ++i) {
        outfile << static_cast<int>(out_arr[i]) << " ";
        if ((i + 1) % cols == 0) {
            outfile << std::endl;
        }
    }
    outfile.close();
}
  


template<int O_CH>
void pool_layer(uint8_t* in_img, uint8_t* out_img, int rows, int cols, const std::string& layer_name) {
    std::cout << "Starting pool_layer for O_CH: " << O_CH << ", rows: " << rows << ", cols: " << cols << std::endl;

    std::vector<uint8_t> in_arr(O_CH * rows * cols);
    std::vector<uint8_t> out_arr(O_CH * (rows / 2) * (cols / 2));

    memcpy(in_arr.data(), in_img, sizeof(uint8_t) * O_CH * rows * cols);
    std::cout << "Copy to in_arr done." << std::endl;

    for (int o_ch = 0; o_ch < O_CH; ++o_ch) {
        //std::cout << "Processing channel " << o_ch << " in pool_layer" << std::endl;
        for (int r = 0; r < rows / 2; ++r) {
            for (int c = 0; c < cols / 2; ++c) {
                uint8_t max_val = 0;
                for (int pr = 0; pr < 2; ++pr) {
                    for (int pc = 0; pc < 2; ++pc) {
                        uint8_t val = in_arr[o_ch * rows * cols + (2 * r + pr) * cols + 2 * c + pc];
                        if (val > max_val) max_val = val;
                    }
                }
                out_arr[o_ch * (rows / 2) * (cols / 2) + r * (cols / 2) + c] = max_val;
            }
        }
    }

    memcpy(out_img, out_arr.data(), sizeof(uint8_t) * O_CH * (rows / 2) * (cols / 2));
    std::ofstream outfile(layer_name + "_output.txt");
    for (int i = 0; i < O_CH * (rows / 2) * (cols / 2); ++i) {
        outfile << static_cast<int>(out_arr[i]) << " ";
        if ((i + 1) % (cols / 2) == 0) {
            outfile << std::endl;
        }
    }
    outfile.close();
}



void fully_connected_layer(float* in_img, float* out_arr, int in_size, int out_size, float* weights, float* bias, const std::string& layer_name) {
    if (!out_arr) {
        std::cerr << "Memory allocation failed in fully_connected_layer" << std::endl;
        return;
    }

    for (int o = 0; o < out_size; ++o) {
        float sum = 0.0;
        for (int i = 0; i < in_size; ++i) {
            sum += in_img[i] * weights[o * in_size + i];
        }
        out_arr[o] = sum + bias[o];
        out_arr[o] = std::max(0.0f, out_arr[o]); // Apply ReLU
    }
    std::ofstream outfile(layer_name + "_output.txt");
    for (int i = 0; i < out_size; ++i) {
        outfile << out_arr[i] << " ";
        if ((i + 1) % 10 == 0) {
            outfile << std::endl;
        }
    }
    outfile.close();
}


void improved_net(uint8_t* input_img, float* fc3_out) {
    std::cout << "Input image first 10 pixels: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(input_img[i]) << " ";
        }
    std::cout << std::endl;
    // Intermediate layers' output
    std::vector<uint8_t> conv1_out(O_CH1 * Rx * Cx);
    std::vector<uint8_t> bn1_out(O_CH1 * Rx * Cx);
    std::vector<uint8_t> pool1_out(O_CH1 * (Rx / 2) * (Cx / 2));
    std::vector<uint8_t> conv2_out(O_CH2 * (Rx / 2) * (Cx / 2));
    std::vector<uint8_t> bn2_out(O_CH2 * (Rx / 2) * (Cx / 2));
    std::vector<uint8_t> pool2_out(O_CH2 * (Rx / 4) * (Cx / 4));
    std::vector<uint8_t> conv3_out(O_CH3 * (Rx / 4) * (Cx / 4));
    std::vector<uint8_t> bn3_out(O_CH3 * (Rx / 4) * (Cx / 4));
    std::vector<uint8_t> pool3_out(O_CH3 * (Rx / 8) * (Cx / 8));
    std::vector<uint8_t> conv4_out(O_CH4 * (Rx / 8) * (Cx / 8));
    std::vector<uint8_t> bn4_out(O_CH4 * (Rx / 8) * (Cx / 8));
    std::vector<uint8_t> pool4_out(O_CH4 * (Rx / 16) * (Cx / 16));
    std::vector<uint8_t> conv5_out(O_CH5 * (Rx / 16) * (Cx / 16));
    std::vector<uint8_t> bn5_out(O_CH5 * (Rx / 16) * (Cx / 16));
    std::vector<uint8_t> pool5_out(O_CH5 * (Rx / 32) * (Cx / 32));
    std::vector<uint8_t> conv6_out(O_CH6 * (Rx / 32) * (Cx / 32));
    std::vector<uint8_t> bn6_out(O_CH6 * (Rx / 32) * (Cx / 32));

    // Convolution Layer 1
    conv_layer<O_CH1, INPUT_CHANNELS, KRx, KCx>(input_img, conv1_out.data(), Rx, Cx, bias1, weight1,"conv1");
    std::cout << "conv1_out[0]: " << (int)conv1_out[0] << std::endl; // �������
    //std::cout << "input_img = " << input_img << std::endl;
    //std::cout << "conv1_out.data() = " << conv1_out.data() << std::endl;
    //std::cout << *bias1 << std::endl;
    //std::cout << "Bias1 values:" << std::endl;
    //for (int i = 0; i < O_CH1; ++i) {
     //   std::cout << bias1[i] << " ";
   // }
   // std::cout << std::endl;

    //std::cout << *weight1 << std::endl;
   // std::cout << "Weight1 values:" << std::endl;
   // for (int o = 0; o < O_CH1; ++o) {
   //     for (int i = 0; i < INPUT_CHANNELS; ++i) {
   //         for (int kr = 0; kr < KRx; ++kr) {
   //             for (int kc = 0; kc < KCx; ++kc) {
   //                 std::cout << static_cast<int>(weight1[o][i][kr][kc]) << " ";
    //            }
   //             std::cout << "|"; // �ָ���ͬ�ľ����˿���
    //        }
   //         std::cout << "  "; // �ָ���ͬ�ľ����˸߶�
  //      }
   //     std::cout << std::endl; // �ָ���ͬ������ͨ��
   // }
    
   // std::cout << "conv1_out[0] = " << conv1_out.data() << std::endl; // �������

    // Batch Normalization Layer 1
    bn_layer<O_CH1>(conv1_out.data(), bn1_out.data(), Rx, Cx, mean1, variance1, gamma1, beta1,"bn1");
    //std::cout << "bn1_out[0]: " << (int)bn1_out[0] << std::endl; // �������
    
    
    //std::cout << "conv1_out.data() = " << conv1_out.data() << std::endl;
    //std::cout << "bn1_out.data() = " << bn1_out.data() << std::endl;

    // Convolution Layer 2
    conv_layer<O_CH2, O_CH1, KRx, KCx>(bn1_out.data(), conv2_out.data(), Rx / 2, Cx / 2, bias2, weight2, "conv2");
    //std::cout << "conv2_out[0]: " << (int)conv2_out[0] << std::endl; // �������
    //std::cout << "bn1_out.data() = " << bn1_out.data() << std::endl;
    //std::cout << "conv2_out.data() = " << conv2_out.data() << std::endl;

    // Batch Normalization Layer 2
    bn_layer<O_CH2>(conv2_out.data(), bn2_out.data(), Rx / 2, Cx / 2, mean2, variance2, gamma2, beta2, "bn2");
    //std::cout << "bn2_out[0]: " << (int)bn2_out[0] << std::endl; // �������
    //std::cout << "conv2_out.data() = " << conv2_out.data() << std::endl;
    //std::cout << "bn2_out.data() = " << bn2_out.data() << std::endl;

    // Pooling Layer 1
    pool_layer<O_CH2>(bn2_out.data(), pool1_out.data(), Rx / 2, Cx / 2, "pool1");
    //std::cout << "pool1_out[0]: " << (int)pool1_out[0] << std::endl; // �������
    //std::cout << "bn2_out.data() = " << bn2_out.data() << std::endl;
    //std::cout << "pool1_out.data() = " << pool1_out.data() << std::endl;

    // Convolution Layer 3
    conv_layer<O_CH3, O_CH2, KRx, KCx>(pool1_out.data(), conv3_out.data(), Rx / 4, Cx / 4, bias3, weight3, "conv3");
   // std::cout << "conv3_out[0]: " << (int)conv3_out[0] << std::endl; // �������
   // std::cout << "pool1_out.data() = " << pool1_out.data() << std::endl;
   // std::cout << "conv3_out.data() = " << conv3_out.data() << std::endl;

    // Batch Normalization Layer 3
    bn_layer<O_CH3>(conv3_out.data(), bn3_out.data(), Rx / 4, Cx / 4, mean3, variance3, gamma3, beta3, "bn3");
   // std::cout << "bn3_out[0]: " << (int)bn3_out[0] << std::endl; // �������
   // std::cout << "conv3_out.data() = " << conv3_out.data() << std::endl;
   // std::cout << "bn3_out.data() = " << bn3_out.data() << std::endl;


    // Convolution Layer 4
    conv_layer<O_CH4, O_CH3, KRx, KCx>(bn3_out.data(), conv4_out.data(), Rx / 8, Cx / 8, bias4, weight4, "conv4");
   // std::cout << "conv4_out[0]: " << (int)conv4_out[0] << std::endl; // �������
 


    // Batch Normalization Layer 4
    bn_layer<O_CH4>(conv4_out.data(), bn4_out.data(), Rx / 8, Cx / 8, mean4, variance4, gamma4, beta4, "bn4");
   // std::cout << "bn4_out[0]: " << (int)bn4_out[0] << std::endl; // �������


    // Pooling Layer 2
    pool_layer<O_CH4>(bn4_out.data(), pool2_out.data(), Rx / 8, Cx / 8, "pool2");
   // std::cout << "pool2_out[0]: " << (int)pool2_out[0] << std::endl; // �������


    // Convolution Layer 5
    conv_layer<O_CH5, O_CH4, KRx, KCx>(pool2_out.data(), conv5_out.data(), Rx / 16, Cx / 16, bias5, weight5, "conv5");
    //std::cout << "conv5_out[0]: " << (int)conv5_out[0] << std::endl; // �������

    // Batch Normalization Layer 5
    bn_layer<O_CH5>(conv5_out.data(), bn5_out.data(), Rx / 16, Cx / 16, mean5, variance5, gamma5, beta5, "bn5");
    //std::cout << "bn5_out[0]: " << (int)bn5_out[0] << std::endl; // �������

    // Convolution Layer 6
    conv_layer<O_CH6, O_CH5, KRx, KCx>(bn5_out.data(), conv6_out.data(), Rx / 32, Cx / 32, bias6, weight6, "conv6");
    //std::cout << "conv6_out[0]: " << (int)conv6_out[0] << std::endl; // �������

    // Batch Normalization Layer 6
    bn_layer<O_CH6>(conv6_out.data(), bn6_out.data(), Rx / 32, Cx / 32, mean6, variance6, gamma6, beta6, "bn6");
    //std::cout << "bn6_out[0]: " << (int)bn6_out[0] << std::endl; // �������

    // Pooling Layer 3
    pool_layer<O_CH6>(bn6_out.data(), pool3_out.data(), Rx / 32, Cx / 32, "pool3");
    //std::cout << "pool3_out[0]: " << (int)pool3_out[0] << std::endl; // �������

    // Allocate memory for fully connected layer inputs
    std::vector<float> fc1_in(O_CH6 * Rx / 32 * Cx / 32);
    std::vector<float> fc1_out(512);
    std::vector<float> fc2_out(256);

    // Flatten the output of the last convolutional layer to feed into the fully connected layer
    for (int i = 0; i < O_CH6 * Rx / 32 * Cx / 32; ++i) {
        fc1_in[i] = static_cast<float>(bn6_out[i]);
        if (i < 10) { // ֻ��ӡǰ10��ֵ���ڵ���
            std::cout << "fc1_in[" << i << "] = " << fc1_in[i] << std::endl; // �������
        }
    }

    fully_connected_layer(fc1_in.data(), fc1_out.data(), O_CH6 * Rx / 32 * Cx / 32, 512, weights_fc1, bias_fc1,"fc1");
    std::cout << "fc1 complete, fc1_out[0]: " << fc1_out[0] << std::endl; // �������
    fully_connected_layer(fc1_out.data(), fc2_out.data(), 512, 256, weights_fc2, bias_fc2, "fc2");
    std::cout << "fc2 complete, fc2_out[0]: " << fc2_out[0] << std::endl; // �������
    fully_connected_layer(fc2_out.data(), fc3_out, 256, 10, weights_fc3, bias_fc3, "fc3");
    std::cout << "fc3 complete, fc3_out[0]: " << fc3_out[0] << std::endl; // �������
}

void load_model_params() {
    std::cout << "Loading parameters for conv1 layer" << std::endl;
    load_and_reshape_weights_new("C:\\damp\\conv1_weight.txt", reinterpret_cast<signed char*>(weight1), O_CH1 * INPUT_CHANNELS * KRx * KCx);
    load_and_reshape_bias("C:\\damp\\conv1_bias.txt", bias1, O_CH1);
    load_and_reshape_bias("C:\\damp\\bn1_running_mean.txt", mean1, O_CH1);
    load_and_reshape_bias("C:\\damp\\bn1_running_var.txt", variance1, O_CH1);
    load_and_reshape_bias("C:\\damp\\bn1_weight.txt", gamma1, O_CH1);
    load_and_reshape_bias("C:\\damp\\bn1_bias.txt", beta1, O_CH1);

    std::cout << "Loading parameters for conv2 layer" << std::endl;
    load_and_reshape_weights_new("C:\\damp\\conv2_weight.txt", reinterpret_cast<signed char*>(weight2), O_CH2 * O_CH1 * KRx * KCx);
    load_and_reshape_bias("C:\\damp\\conv2_bias.txt", bias2, O_CH2);
    load_and_reshape_bias("C:\\damp\\bn2_running_mean.txt", mean2, O_CH2);
    load_and_reshape_bias("C:\\damp\\bn2_running_var.txt", variance2, O_CH2);
    load_and_reshape_bias("C:\\damp\\bn2_weight.txt", gamma2, O_CH2);
    load_and_reshape_bias("C:\\damp\\bn2_bias.txt", beta2, O_CH2);

    std::cout << "Loading parameters for conv3 layer" << std::endl;
    load_and_reshape_weights_new("C:\\damp\\conv3_weight.txt", reinterpret_cast<signed char*>(weight3), O_CH3 * O_CH2 * KRx * KCx);
    load_and_reshape_bias("C:\\damp\\conv3_bias.txt", bias3, O_CH3);
    load_and_reshape_bias("C:\\damp\\bn3_running_mean.txt", mean3, O_CH3);
    load_and_reshape_bias("C:\\damp\\bn3_running_var.txt", variance3, O_CH3);
    load_and_reshape_bias("C:\\damp\\bn3_weight.txt", gamma3, O_CH3);
    load_and_reshape_bias("C:\\damp\\bn3_bias.txt", beta3, O_CH3);

    std::cout << "Loading parameters for conv4 layer" << std::endl;
    load_and_reshape_weights_new("C:\\damp\\conv4_weight.txt", reinterpret_cast<signed char*>(weight4), O_CH4 * O_CH3 * KRx * KCx);
    load_and_reshape_bias("C:\\damp\\conv4_bias.txt", bias4, O_CH4);
    load_and_reshape_bias("C:\\damp\\bn4_running_mean.txt", mean4, O_CH4);
    load_and_reshape_bias("C:\\damp\\bn4_running_var.txt", variance4, O_CH4);
    load_and_reshape_bias("C:\\damp\\bn4_weight.txt", gamma4, O_CH4);
    load_and_reshape_bias("C:\\damp\\bn4_bias.txt", beta4, O_CH4);

    std::cout << "Loading parameters for conv5 layer" << std::endl;
    load_and_reshape_weights_new("C:\\damp\\conv5_weight.txt", reinterpret_cast<signed char*>(weight5), O_CH5 * O_CH4 * KRx * KCx);
    load_and_reshape_bias("C:\\damp\\conv5_bias.txt", bias5, O_CH5);
    load_and_reshape_bias("C:\\damp\\bn5_running_mean.txt", mean5, O_CH5);
    load_and_reshape_bias("C:\\damp\\bn5_running_var.txt", variance5, O_CH5);
    load_and_reshape_bias("C:\\damp\\bn5_weight.txt", gamma5, O_CH5);
    load_and_reshape_bias("C:\\damp\\bn5_bias.txt", beta5, O_CH5);

    std::cout << "Loading parameters for conv6 layer" << std::endl;
    load_and_reshape_weights_new("C:\\damp\\conv6_weight.txt", reinterpret_cast<signed char*>(weight6), O_CH6 * O_CH5 * KRx * KCx);
    load_and_reshape_bias("C:\\damp\\conv6_bias.txt", bias6, O_CH6);
    load_and_reshape_bias("C:\\damp\\bn6_running_mean.txt", mean6, O_CH6);
    load_and_reshape_bias("C:\\damp\\bn6_running_var.txt", variance6, O_CH6);
    load_and_reshape_bias("C:\\damp\\bn6_weight.txt", gamma6, O_CH6);
    load_and_reshape_bias("C:\\damp\\bn6_bias.txt", beta6, O_CH6);

    std::cout << "Loading parameters for fully connected layers" << std::endl;
    load_and_reshape_bias("C:\\damp\\fc1_weight.txt", weights_fc1, 512 * (O_CH6 * Rx / 32 * Cx / 32));
    load_and_reshape_bias("C:\\damp\\fc1_bias.txt", bias_fc1, 512);
    load_and_reshape_bias("C:\\damp\\fc2_weight.txt", weights_fc2, 256 * 512);
    load_and_reshape_bias("C:\\damp\\fc2_bias.txt", bias_fc2, 256);
    load_and_reshape_bias("C:\\damp\\fc3_weight.txt", weights_fc3, 10 * 256);
    load_and_reshape_bias("C:\\damp\\fc3_bias.txt", bias_fc3, 10);
    std::cout << "111111";
    std::cout << "Model parameters loaded." << std::endl;
    std::cout << "222222";
    // ��ӡȫ���Ӳ�������ڵ���
    std::cout << "FC1 weights: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << weights_fc1[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "FC1 bias: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << bias_fc1[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "FC2 weights: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << weights_fc2[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "FC2 bias: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << bias_fc2[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "FC3 weights: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << weights_fc3[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "FC3 bias: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << bias_fc3[i] << " ";
    }
    std::cout << std::endl;
}

// CIFAR-10 data loader
struct Image {
    uint8_t data[3 * 32 * 32];
    uint8_t label;
};

std::vector<Image> load_cifar10(const std::string& batch_file) {
    std::ifstream file(batch_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CIFAR-10 batch file " << batch_file << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<Image> images;
    Image img;
    while (file.read(reinterpret_cast<char*>(&img.label), 1)) {
        file.read(reinterpret_cast<char*>(img.data), 3 * 32 * 32);
        images.push_back(img);
        //file.close();
    }
    

    file.close();
   // std::cout << "First image first 10 pixels: ";
   // for (int i = 0; i < 10; ++i) {
   //     std::cout << static_cast<int>(images[0].data[i]) << " ";
   // }
   // std::cout << std::endl;
    return images;
}


// Evaluate model accuracy on CIFAR-10
//void evaluate_model(const std::vector<Image>& test_images) {
    //int correct = 0;
    //float fc3_out[10] = {};
    //std::cout << "Number of test images: " << test_images.size() << std::endl;
    //int i = 0;
    //for (const auto& img : test_images) {
    //    improved_net(const_cast<uint8_t*>(img.data), fc3_out);
    //    int predicted = std::max_element(fc3_out, fc3_out + 10) - fc3_out;

    //    i++;
    //    std::cout << "Evaluating image: " << i << std::endl;
    //    if (predicted == img.label) {
    //        ++correct;
    //    }
   // }

    //float accuracy = 100.0f * correct / test_images.size();
    //std::cout << std::endl << "Test Accuracy: " << accuracy << "%" << std::endl;
//}
void evaluate_model(const std::vector<Image>& test_images, int batch_size) {
    int correct = 0;
    float fc3_out[10] = {};
    std::cout << "Number of test images: " << test_images.size() << std::endl;
    int total_batches = test_images.size() / batch_size;

    for (int batch = 0; batch < total_batches; ++batch) {
        for (int i = 0; i < batch_size; ++i) {
            const auto& img = test_images[batch * batch_size + i];
            improved_net(const_cast<uint8_t*>(img.data), fc3_out);
            int predicted = std::max_element(fc3_out, fc3_out + 10) - fc3_out;

            std::cout << "Evaluating image: " << (batch * batch_size + i + 1) << std::endl;
            if (predicted == img.label) {
                ++correct;
            }
        }
    }

    float accuracy = 100.0f * correct / (total_batches * batch_size);
    std::cout << std::endl << "Test Accuracy: " << accuracy << "%" << std::endl;
}

void test_simple_input() {
    uint8_t simple_input[3 * 32 * 32];
    std::fill_n(simple_input, 3 * 32 * 32, 1);  // All ones

    float test_bias[32];
    std::fill_n(test_bias, 32, 1.0f);  // All ones

    int8_t test_weight[32][3][3][3];
    std::fill(&test_weight[0][0][0][0], &test_weight[0][0][0][0] + 32 * 3 * 3 * 3, 1);  // All ones

    uint8_t test_output[32 * 32 * 32];
    conv_layer<32, 3, 3, 3>(simple_input, test_output, 32, 32, test_bias, test_weight, "test");

    std::cout << "Test output: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(test_output[i]) << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Load model parameters
    load_model_params();

    // Load CIFAR-10 test dataset
    std::string cifar10_test_batch = "D:\\cifar10\\cifar-10-batches-bin\\test_batch.bin"; // Ensure this path points to the correct CIFAR-10 test batch file
    std::vector<Image> test_images = load_cifar10(cifar10_test_batch);
    int batch_size = 5;

    // Evaluate model on CIFAR-10 test dataset
    evaluate_model(test_images, batch_size);

    // Test with simple input
    test_simple_input();

    return 0;
}
