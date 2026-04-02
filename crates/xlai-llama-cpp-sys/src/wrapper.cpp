#include "json-schema-to-grammar.h"
#include "sampling.h"

#include <nlohmann/json.hpp>

#include <exception>
#include <string>

namespace {

thread_local std::string g_last_error;

void set_last_error(std::string message) {
    g_last_error = std::move(message);
}

void clear_last_error() {
    g_last_error.clear();
}

} // namespace

extern "C" const char * xlai_llama_last_error_message() {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

extern "C" struct llama_sampler * xlai_llama_sampler_init_llguidance(
    const struct llama_vocab * vocab,
    const char * grammar_kind,
    const char * grammar_data
) {
    clear_last_error();

    if (vocab == nullptr) {
        set_last_error("llama.cpp LLGuidance sampler requires a non-null vocab pointer");
        return nullptr;
    }

    if (grammar_kind == nullptr || grammar_data == nullptr) {
        set_last_error("llama.cpp LLGuidance sampler requires non-null grammar inputs");
        return nullptr;
    }

    struct llama_sampler * sampler = llama_sampler_init_llg(vocab, grammar_kind, grammar_data);
    if (sampler == nullptr) {
        set_last_error("llama.cpp could not create the LLGuidance sampler");
    }
    return sampler;
}

extern "C" struct llama_sampler * xlai_llama_sampler_init_json_schema(
    const struct llama_vocab * vocab,
    const char * json_schema
) {
    clear_last_error();

    if (vocab == nullptr) {
        set_last_error("llama.cpp JSON schema sampler requires a non-null vocab pointer");
        return nullptr;
    }

    if (json_schema == nullptr) {
        set_last_error("llama.cpp JSON schema sampler requires a non-null schema string");
        return nullptr;
    }

    try {
        const auto schema = nlohmann::ordered_json::parse(json_schema);
        const auto grammar = json_schema_to_grammar(schema, false);

        struct llama_sampler * sampler = nullptr;
        if (grammar.rfind("%llguidance", 0) == 0) {
            sampler = xlai_llama_sampler_init_llguidance(vocab, "lark", grammar.c_str());
            if (sampler == nullptr) {
                set_last_error("llama.cpp could not create the LLGuidance sampler for the JSON schema");
            }
            return sampler;
        }

        sampler = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
        if (sampler == nullptr) {
            set_last_error("llama.cpp could not create a grammar sampler for the JSON schema");
        }
        return sampler;
    } catch (const std::exception & error) {
        set_last_error(error.what());
        return nullptr;
    }
}
