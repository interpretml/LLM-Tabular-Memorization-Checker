import tabmemcheck as tabmem


class TestLLM(tabmem.LLM_Interface):
    def completion(self, prompt, temperature, max_tokens):
        return "THIS IS A TEST."

    def chat_completion(self, messages, temperature, max_tokens):
        """Returns: The response (string)"""
        return "THIS IS A TEST."
