// L8 reference: the plugin's ring KV must equal llama.cpp's --ctx-shift state.
//
// Feed a token list longer than n_ctx by raw llama_decode (llama-cli would refuse/truncate):
// decode ids[0..n_ctx-1] at absolute positions, llama_memory_seq_rm the oldest shift tokens
// (the ctx-shift primitive), decode the rest, then dump the logits row of the last position.
// The visible cell set is then exactly the last n_ctx positions -- the same set the plugin's
// ring attends -- so the logits rows must agree (cosine ~1, same argmax).
//
// usage: ring_ref <model.gguf> <n_ctx> <ids.txt> <out.logits>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " <model.gguf> <n_ctx> <ids.txt> <out.logits>\n";
        return 1;
    }
    const int n_ctx = std::atoi(argv[2]);
    std::ifstream f(argv[3]);
    std::vector<llama_token> ids;
    for (int id; f >> id;) {
        ids.push_back(id);
    }
    if (ids.empty()) {
        std::cerr << "no ids in " << argv[3] << "\n";
        return 1;
    }
    const size_t T = ids.size();
    const size_t shift = T > static_cast<size_t>(n_ctx) ? T - n_ctx : 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::cerr << "failed to load " << argv[1] << "\n";
        return 2;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // match the plugin's dumped graphs
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::cerr << "failed to create context\n";
        return 3;
    }

    // First decode: fill the cache (positions 0..min(T, n_ctx)-1). When the sequence fits
    // (no shift), the last row carries the logits.
    const size_t n_first = std::min<size_t>(T, n_ctx);
    llama_batch batch = llama_batch_init(n_first, 0, 1);
    for (size_t i = 0; i < n_first; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = shift == 0 && i + 1 == n_first ? 1 : 0;
    }
    batch.n_tokens = n_first;
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "first decode failed\n";
        return 4;
    }

    // ctx-shift: decode the rest in <= n_batch chunks, freeing exactly n oldest positions
    // before each chunk (the server's shift does the same).
    size_t last_rows = n_first;
    if (shift > 0) {
        size_t done = 0;
        while (done < shift) {
            const size_t n = std::min<size_t>(shift - done, n_ctx);
            // free the n oldest live positions: after `done` shifted tokens they sit at
            // done .. done + n
            llama_memory_seq_rm(llama_get_memory(ctx), 0, static_cast<llama_pos>(done),
                                static_cast<llama_pos>(done + n));
            llama_batch rest = llama_batch_init(n, 0, 1);
            for (size_t i = 0; i < n; ++i) {
                rest.token[i] = ids[n_ctx + done + i];
                rest.pos[i] = static_cast<llama_pos>(n_ctx + done + i);
                rest.n_seq_id[i] = 1;
                rest.seq_id[i][0] = 0;
                rest.logits[i] = done + i + 1 == shift ? 1 : 0;
            }
            rest.n_tokens = n;
            if (llama_decode(ctx, rest) != 0) {
                std::cerr << "shifted decode failed\n";
                return 5;
            }
            done += n;
            last_rows = n;
        }
    }

    // Dump the logits row of the last decoded row (logits are indexed by batch row, not
    // position: the last chunk's last row).
    const float* logits = llama_get_logits_ith(ctx, static_cast<int32_t>(last_rows - 1));
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    {
        std::ofstream o(argv[4], std::ios::binary);
        o.write(reinterpret_cast<const char*>(logits), static_cast<size_t>(vocab) * sizeof(float));
    }
    size_t argmax = 0;
    for (size_t i = 1; i < static_cast<size_t>(vocab); ++i) {
        if (logits[i] > logits[argmax]) {
            argmax = i;
        }
    }
    std::cout << "n_ctx " << n_ctx << ", ids " << T << ", shifted " << shift << ", vocab " << vocab
              << ", argmax " << argmax << "\n";

    // GEN=<n> (env): greedy-continue n tokens from the same state, printing each token id, so a
    // driver run's generated tokens can be compared against llama.cpp's ctx-shift continuation.
    if (const char* gen = std::getenv("GEN")) {
        const int G = std::atoi(gen);
        llama_batch one = llama_batch_init(1, 0, 1);
        for (int g = 0; g < G; ++g) {
            // free the oldest live position (shift + g) before each decode
            llama_memory_seq_rm(llama_get_memory(ctx), 0, static_cast<llama_pos>(shift + g),
                                static_cast<llama_pos>(shift + g + 1));
            one.token[0] = static_cast<llama_token>(argmax);
            one.pos[0] = static_cast<llama_pos>(T + g);
            one.n_seq_id[0] = 1;
            one.seq_id[0][0] = 0;
            one.logits[0] = 1;
            one.n_tokens = 1;
            if (llama_decode(ctx, one) != 0) {
                std::cerr << "greedy decode failed at step " << g << "\n";
                return 6;
            }
            const float* next = llama_get_logits_ith(ctx, 0);
            argmax = 0;
            for (size_t i = 1; i < static_cast<size_t>(vocab); ++i) {
                if (next[i] > next[argmax]) {
                    argmax = i;
                }
            }
            std::cout << argmax << " ";
        }
        std::cout << "\n";
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
