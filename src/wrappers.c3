// wrappers.c3
// Thin, optional convenience layer on top of the raw llama.h bindings.
// The raw bindings (*.c3i) stay complete and direct; these wrappers only
// cover the handful of spots where the raw C API is unusually tedious to
// call correctly from C3: the tokenize/detokenize size-probing dance, the
// batch/decode loop, and null/negative-return error handling. Everything
// else is already a clean 1:1 call - use the raw `llama::` functions
// directly for that.
//
// These live in their own submodules (llama::model, llama::context,
// llama::vocab) so names like `load`/`decode`/`tokenize` don't collide
// with the raw top-level `llama::model_load_from_file`/`llama::decode`/
// `llama::tokenize` functions they call into.
//
// Memory: every LlamaToken[]/String these functions return is allocated
// with the passed-in `allocator` (the thread heap by default) and is the
// caller's to free - e.g. `allocator.release(tokens.ptr, false)`, or pass
// `tmem` and wrap the call site in `@pool() { ... };` to avoid manual
// frees entirely (see example/generate.c3).

module llama::model;

faultdef LOAD_FAILED;

// Load a model from a file, raising LOAD_FAILED instead of returning null.
fn LlamaModel*? load(String path, LlamaModelParams params = llama::model_default_params())
{
	LlamaModel* model = llama::model_load_from_file(path.zstr_tcopy(), params);
	return model ? model : LOAD_FAILED~;
}

module llama::context;

faultdef INIT_FAILED, DECODE_FAILED, ENCODE_FAILED;

// Create a context for a model, raising INIT_FAILED instead of returning null.
fn LlamaContext*? init(LlamaModel* model, LlamaContextParams params = llama::context_default_params())
{
	if (!model) return INIT_FAILED~;
	LlamaContext* ctx = llama::init_from_model(model, params);
	return ctx ? ctx : INIT_FAILED~;
}

// Decode a run of tokens for the default sequence, without having to build
// a LlamaBatch by hand. Wraps llama_batch_get_one(), which does not need a
// matching batch_free() call.
//
// llama_decode()'s raw return code distinguishes KV-cache-full (1),
// user-abort (2), and fatal (< -1) cases - this wrapper collapses all of
// them to DECODE_FAILED. Call llama::decode() directly if you need that
// distinction.
fn void? decode(LlamaContext* ctx, LlamaToken[] tokens)
{
	if (!ctx) return DECODE_FAILED~;
	if (!tokens.len) return;
	LlamaBatch batch = llama::batch_get_one(tokens.ptr, (int)tokens.len);
	if (llama::decode(ctx, batch) != 0) return DECODE_FAILED~;
}

// Encoder-side equivalent of decode(), for encoder-decoder models.
fn void? encode(LlamaContext* ctx, LlamaToken[] tokens)
{
	if (!ctx) return ENCODE_FAILED~;
	if (!tokens.len) return;
	LlamaBatch batch = llama::batch_get_one(tokens.ptr, (int)tokens.len);
	if (llama::encode(ctx, batch) != 0) return ENCODE_FAILED~;
}

module llama::vocab;

import std::core::mem;

faultdef TOKENIZE_FAILED, DETOKENIZE_FAILED, TOKEN_TO_PIECE_FAILED;

// Tokenize text into a freshly allocated token array (freed by the caller).
// Hides the two-call "ask for the size, then ask again with a buffer of
// that size" dance the raw llama_tokenize() requires.
fn LlamaToken[]? tokenize(LlamaVocab* vocab, String text, bool add_special, bool parse_special, Allocator allocator = mem)
{
	int n = llama::tokenize(vocab, text.ptr, (int)text.len, null, 0, add_special, parse_special);
	// llama_tokenize() returns INT32_MIN (not just "negative") on an
	// internal overflow - negating it would overflow right back to
	// itself in two's complement, so it must be checked before `-n`.
	if (n == int::min) return TOKENIZE_FAILED~;
	int needed = n < 0 ? -n : n;
	LlamaToken[] tokens = mem::alloc::new_array(allocator, LlamaToken, needed);
	if (needed == 0) return tokens;
	int actual = llama::tokenize(vocab, text.ptr, (int)text.len, tokens.ptr, needed, add_special, parse_special);
	if (actual < 0)
	{
		allocator.release(tokens.ptr, false);
		return TOKENIZE_FAILED~;
	}
	return tokens[:actual];
}

// Detokenize a token array back into text (inverse of tokenize()).
fn String? detokenize(LlamaVocab* vocab, LlamaToken[] tokens, bool remove_special, bool unparse_special, Allocator allocator = mem)
{
	int n = llama::detokenize(vocab, tokens.ptr, (int)tokens.len, null, 0, remove_special, unparse_special);
	if (n == int::min) return DETOKENIZE_FAILED~;
	int needed = n < 0 ? -n : n;
	char[] buf = mem::alloc::new_array(allocator, char, needed);
	if (needed == 0) return (String)buf;
	int actual = llama::detokenize(vocab, tokens.ptr, (int)tokens.len, buf.ptr, needed, remove_special, unparse_special);
	if (actual < 0)
	{
		allocator.release(buf.ptr, false);
		return DETOKENIZE_FAILED~;
	}
	return (String)buf[:actual];
}

// Render a single token as text (e.g. to print a model's output as it
// generates, one token at a time).
fn String? token_to_piece(LlamaVocab* vocab, LlamaToken token, bool special, Allocator allocator = mem)
{
	int n = llama::token_to_piece(vocab, token, null, 0, 0, special);
	if (n == int::min) return TOKEN_TO_PIECE_FAILED~;
	int needed = n < 0 ? -n : n;
	char[] buf = mem::alloc::new_array(allocator, char, needed);
	if (needed == 0) return (String)buf;
	int actual = llama::token_to_piece(vocab, token, buf.ptr, needed, 0, special);
	if (actual < 0)
	{
		allocator.release(buf.ptr, false);
		return TOKEN_TO_PIECE_FAILED~;
	}
	return (String)buf[:actual];
}
