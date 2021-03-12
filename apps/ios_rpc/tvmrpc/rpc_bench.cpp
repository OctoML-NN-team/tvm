//
//  rpc_bench.cpp
//  tvmrpc
//
//  Created by Alexander Peskov on 11.03.2021.
//  Copyright © 2021 dmlc. All rights reserved.
//

#include "rpc_bench.h"

#include "rpc_bench.h"
#include <stddef.h>
#include <chrono>
#include <iostream>

int g_num = 13;

float g_answ;

using Time = std::chrono::high_resolution_clock;
using mks = std::chrono::microseconds;

std::vector<float> ma;
std::vector<float> mb;
std::vector<float> mc;


void bench_impl(int num) {
    size_t size = 2 << num;
    float a[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float b[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float abs = 0;
    float sign = 1;
    
    const int vec_size = 8;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < vec_size; j++) {
            a[j] += b[j];
        }
        abs++;
        sign *= -1;
        for (int j = 0; j < vec_size; j++) {
            b[j] = abs * sign;
        }
    }
    g_answ = 0;
    for (float el : a)
        g_answ += el;
}

int bench_wrp() {
    
    auto start = Time::now();
    
    bench_impl(g_num);
    
    auto dur = Time::now() - start;
    auto dur_mks = std::chrono::duration_cast<mks>(dur).count();
    
    return dur_mks;
}

namespace tvm {
namespace runtime {

int bench() {
    const int num_exp = 64;
    const int min = 700;
    
    int stat[num_exp] = {};
    int mean = 0;
    
    for (int i = 0; i < num_exp; i++) {
        auto score = bench_wrp();
        mean += score - min;
        stat[i] = score;
//        if (score >= min + num_exp)
//            stat[num_exp - 1]++;
//        else
//            stat[score - min]++;
    }
    mean /= num_exp;

    std::sort(stat, stat + num_exp);
    return stat[num_exp/2];
//    int max_pos = 0;
//    for (int i = 0; i < num_exp; i++) {
//        if (stat[max_pos] < stat[i]) {
//            max_pos = i;
//        }
//    }
//    for (int i = std::max(max_pos - 5, 0); i < std::min(max_pos + 5, num_exp); i++) {
//        std::cout << " " << (int)stat[i];
//    }
//    std::cout << std::endl;
//
//    return max_pos + min;
}

void simple_bench(int num_op) {
  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::nanoseconds ns;
  
  num_op = 512;
  
  int M = num_op;
  int N = num_op;
  int K = num_op;
  float alfa =1.0;
  float beta =0.0;
    auto update_matrix = [] (std::vector<float> &mat, int size, float val) {
        if (mat.size() != size)
            mat.resize(size, val);
    };
    update_matrix(ma, M*N, 0.5);
    update_matrix(mb, N*K, 2);
    update_matrix(mc, M*K, 0.0);

  Time::time_point start = Time::now();
    int num = 20;
    for (int i =0; i < num; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    alfa,
                    ma.data(), M,
                    mb.data(), N,
                    beta,
                    mc.data(), M
        );
    }

  auto dur = Time::now() - start;
  
  std::cout << "[BENCH] time: " << std::chrono::duration_cast<ns>(dur).count() / num * 0.001 << " mks" << std::endl;
}

}
}
