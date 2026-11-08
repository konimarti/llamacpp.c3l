# llamacpp.c3l

C3 bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) - run
GGUF language models directly from C3.

### Features

- Full coverage of `llama.h`'s public API: model loading, context, batching,
  tokenization, sampling, LoRA adapters, quantization, and training.
- Direct binding - C3 names mirror the C names, so anything you know from
  the C API or the official docs carries straight over.
- Convenience wrappers for the few spots where the raw C API is tedious:
  tokenizing, batching/decoding, and error handling via C3 optionals.
- No C3 dependencies - just the native `libllama`/`libggml` libraries.

### Why

You want llama.cpp's inference engine in a C3 program without writing or
maintaining the FFI glue yourself.

### Installation

You'll need [c3c](https://c3-lang.org) installed, and a GGUF model file
(a quantized model format llama.cpp reads directly - download one from
e.g. [Hugging Face](https://huggingface.co/models?library=gguf)).

1. Build llama.cpp as shared libraries: `cmake -B llamacpp-build -DBUILD_SHARED_LIBS=ON && cmake --build llamacpp-build` from
   [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), then install
   or point your linker (and `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` at
   runtime) at the resulting `libllama`/`libggml`/`libggml-base`/`libggml-cpu`.
2. In your project's `project.json`, add this library's path to
   `dependency-search-paths` and add `llama` to `dependencies`.

### Example

```c3
import llama;

fn void main()
{
    llama::backend_init();
    defer llama::backend_free();

    LlamaModel* model = llama::model::load("model.gguf")!!;
    defer llama::model_free(model);

    LlamaVocab* vocab = llama::model_get_vocab(model);

    LlamaContext* ctx = llama::context::init(model)!!;
    defer llama::free(ctx);

    @pool() // tmem below is freed automatically when this block ends
    {
        LlamaToken[] tokens = llama::vocab::tokenize(vocab, "Hello,", true, true, tmem)!!;
        llama::context::decode(ctx, tokens)!!;
    };
}
```

A full generate-text-from-a-prompt example is in
[example/generate.c3](example/generate.c3) - run it with
`c3c run generate -- path/to/model.gguf "your prompt"`.
