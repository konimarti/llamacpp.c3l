<*
 Tests for the llama::model/llama::context/llama::vocab convenience
 wrappers in src/wrappers.c3. These link against the fake stub library in
 .github/ci-stub/ (build it first - see README.md or .github/workflows/ci.yml
 for the exact commands), not a real libllama, so they run fast and
 deterministically in CI.
*>
module llama_wrappers_test @test;

import llama;
import std::io;

fn void test_model_load_and_context_init() => @pool()
{
	llama::backend_init();
	defer llama::backend_free();

	LlamaModel* model = llama::model::load("fake.gguf")!!;
	defer llama::model_free(model);

	LlamaContext* ctx = llama::context::init(model)!!;
	defer llama::free(ctx);

	assert(model != null);
	assert(ctx != null);
}

fn void test_context_init_rejects_null_model()
{
	if (catch err = llama::context::init(null))
	{
		assert(err == llama::context::INIT_FAILED, "got: %s", err);
		return;
	}
	assert(false, "expected INIT_FAILED for a null model");
}

fn void test_decode_rejects_null_context()
{
	LlamaToken[1] tokens = { 1 };
	if (catch err = llama::context::decode(null, tokens[..]))
	{
		assert(err == llama::context::DECODE_FAILED, "got: %s", err);
		return;
	}
	assert(false, "expected DECODE_FAILED for a null context");
}

fn void test_decode_accepts_empty_token_slice() => @pool()
{
	llama::backend_init();
	defer llama::backend_free();

	LlamaModel* model = llama::model::load("fake.gguf")!!;
	defer llama::model_free(model);

	LlamaContext* ctx = llama::context::init(model)!!;
	defer llama::free(ctx);

	// An empty slice must be a no-op success, not forwarded into
	// llama_batch_get_one() with n_tokens == 0.
	LlamaToken[] empty;
	llama::context::decode(ctx, empty)!!;
}

fn void test_tokenize_detokenize_round_trip() => @pool()
{
	llama::backend_init();
	defer llama::backend_free();

	LlamaModel* model = llama::model::load("fake.gguf")!!;
	defer llama::model_free(model);

	LlamaVocab* vocab = llama::model_get_vocab(model);

	LlamaToken[] tokens = llama::vocab::tokenize(vocab, "hello", true, true, tmem)!!;
	assert(tokens.len > 0, "expected at least one token for non-empty text");

	String text = llama::vocab::detokenize(vocab, tokens, false, false, tmem)!!;
	assert(text.len > 0, "expected non-empty detokenized text");
}

fn void test_tokenize_empty_text_returns_empty_slice() => @pool()
{
	llama::backend_init();
	defer llama::backend_free();

	LlamaModel* model = llama::model::load("fake.gguf")!!;
	defer llama::model_free(model);

	LlamaVocab* vocab = llama::model_get_vocab(model);

	LlamaToken[] tokens = llama::vocab::tokenize(vocab, "", true, true, tmem)!!;
	assert(tokens.len == 0, "expected zero tokens for empty text, got %d", tokens.len);
}

fn void test_token_to_piece() => @pool()
{
	llama::backend_init();
	defer llama::backend_free();

	LlamaModel* model = llama::model::load("fake.gguf")!!;
	defer llama::model_free(model);

	LlamaVocab* vocab = llama::model_get_vocab(model);

	String piece = llama::vocab::token_to_piece(vocab, 1, true, tmem)!!;
	assert(piece.len > 0, "expected a non-empty piece");
}
