#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "llama.h"

// Static dummy types for opaque handles
struct llama_model {
    int dummy;
};

struct llama_vocab {
    int dummy;
};

struct llama_context {
    int dummy;
};

// Static dummy objects - no malloc/free used
static struct llama_model   s_dummy_model;
static struct llama_vocab   s_dummy_vocab;
static struct llama_context s_dummy_context;
static struct llama_sampler s_dummy_sampler_chain;
static struct llama_sampler s_dummy_sampler_greedy;

#define STUB_TOKEN_EOG 999

static const char * token_piece(llama_token token) {
    switch (token) {
        case 1:   return " Paris";
        case 2:   return " is";
        case 3:   return " great.";
        case 100: return " The";
        case 101: return " capital";
        case 102: return " of";
        case 103: return " France";
        default:  return " test";
    }
}

// Backend init / free
LLAMA_API void llama_backend_init(void) {
}

LLAMA_API void llama_backend_free(void) {
}

// Default parameters
LLAMA_API struct llama_model_params llama_model_default_params(void) {
    struct llama_model_params p;
    memset(&p, 0, sizeof(p));
    return p;
}

LLAMA_API struct llama_context_params llama_context_default_params(void) {
    struct llama_context_params p;
    memset(&p, 0, sizeof(p));
    p.n_ctx = 2048;
    p.n_batch = 512;
    p.n_ubatch = 512;
    return p;
}

LLAMA_API struct llama_sampler_chain_params llama_sampler_chain_default_params(void) {
    struct llama_sampler_chain_params p;
    memset(&p, 0, sizeof(p));
    return p;
}

// Model & Context lifecycle
LLAMA_API struct llama_model * llama_model_load_from_file(
                         const char * path_model,
          struct llama_model_params   params) {
    (void)path_model;
    (void)params;
    return &s_dummy_model;
}

LLAMA_API void llama_model_free(struct llama_model * model) {
    (void)model;
}

LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model) {
    (void)model;
    return &s_dummy_vocab;
}

LLAMA_API struct llama_context * llama_init_from_model(
                 struct llama_model * model,
        struct llama_context_params   params) {
    (void)model;
    (void)params;
    return &s_dummy_context;
}

LLAMA_API void llama_free(struct llama_context * ctx) {
    (void)ctx;
}

// Batch & Decode/Encode
LLAMA_API struct llama_batch llama_batch_get_one(
              llama_token * tokens,
                  int32_t   n_tokens) {
    struct llama_batch batch;
    memset(&batch, 0, sizeof(batch));
    batch.n_tokens = n_tokens;
    batch.token = tokens;
    return batch;
}

LLAMA_API int32_t llama_decode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    (void)ctx;
    (void)batch;
    return 0;
}

LLAMA_API int32_t llama_encode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    (void)ctx;
    (void)batch;
    return 0;
}

// Tokenize, Detokenize, Token-to-Piece (two-call size-probing pattern)
LLAMA_API int32_t llama_tokenize(
    const struct llama_vocab * vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    (void)vocab;
    (void)text;
    (void)add_special;
    (void)parse_special;
    int32_t needed = (text != NULL && text_len > 0) ? 4 : 0;
    if (tokens == NULL || n_tokens_max < needed) {
        return -needed;
    }
    for (int32_t i = 0; i < needed; i++) {
        tokens[i] = 100 + i;
    }
    return needed;
}

LLAMA_API int32_t llama_detokenize(
    const struct llama_vocab * vocab,
           const llama_token * tokens,
                     int32_t   n_tokens,
                        char * text,
                     int32_t   text_len_max,
                        bool   remove_special,
                        bool   unparse_special) {
    (void)vocab;
    (void)remove_special;
    (void)unparse_special;
    int32_t total_len = 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        total_len += (int32_t)strlen(token_piece(tokens[i]));
    }
    if (text == NULL || text_len_max < total_len) {
        return -total_len;
    }
    int32_t pos = 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        const char * p = token_piece(tokens[i]);
        int32_t plen = (int32_t)strlen(p);
        memcpy(text + pos, p, (size_t)plen);
        pos += plen;
    }
    return total_len;
}

LLAMA_API int32_t llama_token_to_piece(
        const struct llama_vocab * vocab,
                     llama_token   token,
                            char * buf,
                         int32_t   length,
                         int32_t   lstrip,
                            bool   special) {
    (void)vocab;
    (void)special;
    const char * piece = token_piece(token);
    int32_t len = (int32_t)strlen(piece);
    while (lstrip > 0 && len > 0 && *piece == ' ') {
        piece++;
        len--;
        lstrip--;
    }
    if (buf == NULL || length < len) {
        return -len;
    }
    memcpy(buf, piece, (size_t)len);
    return len;
}

// Samplers
LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params) {
    (void)params;
    return &s_dummy_sampler_chain;
}

LLAMA_API void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl) {
    (void)chain;
    (void)smpl;
}

LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void) {
    return &s_dummy_sampler_greedy;
}

LLAMA_API void llama_sampler_free(struct llama_sampler * smpl) {
    (void)smpl;
}

static int s_sample_count = 0;

LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    (void)smpl;
    (void)ctx;
    (void)idx;
    // Return 3 regular tokens (1, 2, 3), then EOG (999) to exercise loop and exit path
    static const llama_token seq[] = { 1, 2, 3, STUB_TOKEN_EOG };
    int n_seq = (int)(sizeof(seq) / sizeof(seq[0]));
    llama_token t = seq[s_sample_count % n_seq];
    s_sample_count++;
    return t;
}

LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token) {
    (void)vocab;
    return token == STUB_TOKEN_EOG;
}
