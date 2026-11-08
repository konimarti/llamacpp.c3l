// Minimal end-to-end example: load a local GGUF model and generate text
// from a prompt. Get a model first, e.g.:
//   curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
//
// Build and run:
//   c3c run generate -- path/to/model.gguf "The capital of France is"
module generate;

import llama;
import std::io;

fn void main(String[] args)
{
	if (args.len < 2)
	{
		io::printfn("usage: %s <model.gguf> [prompt]", args.len > 0 ? args[0] : "generate");
		return;
	}

	String model_path = args[1];
	String prompt = args.len > 2 ? args[2] : "The capital of France is";

	llama::backend_init();
	defer llama::backend_free();

	LlamaModel* model = llama::model::load(model_path)!!;
	defer llama::model_free(model);

	LlamaVocab* vocab = llama::model_get_vocab(model);

	LlamaContextParams ctx_params = llama::context_default_params();
	ctx_params.n_ctx = 2048;
	LlamaContext* ctx = llama::context::init(model, ctx_params)!!;
	defer llama::free(ctx);

	LlamaSampler* sampler = llama::sampler_chain_init(llama::sampler_chain_default_params());
	llama::sampler_chain_add(sampler, llama::sampler_init_greedy());
	defer llama::sampler_free(sampler);

	// @pool()/tmem: tokenize() and token_to_piece() below allocate with the
	// temporary allocator, freed automatically when this block ends -
	// no manual frees needed.
	@pool()
	{
		LlamaToken[] prompt_tokens = llama::vocab::tokenize(vocab, prompt, true, true, tmem)!!;

		int n_predict = 64;
		if (prompt_tokens.len + n_predict > ctx_params.n_ctx)
		{
			io::printfn("prompt too long for n_ctx=%d - shorten it or raise n_ctx", ctx_params.n_ctx);
			return;
		}

		llama::context::decode(ctx, prompt_tokens)!!;

		io::printf("%s", prompt);
		for (int i = 0; i < n_predict; i++)
		{
			LlamaToken token = llama::sampler_sample(sampler, ctx, -1);
			if (token < 0 || llama::vocab_is_eog(vocab, token)) break;

			io::printf("%s", llama::vocab::token_to_piece(vocab, token, true, tmem)!!);

			// Only feed the token back in if another sampling round will
			// actually use it - the last iteration's token never gets used.
			if (i + 1 < n_predict)
			{
				LlamaToken[1] next = { token };
				llama::context::decode(ctx, next[..])!!;
			}
		}
		io::printn("");
	};
}
