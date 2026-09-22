# AI-Use Appendix - Ritika Mukesh Neema

1. **Which parts did you use an assistant for, and which did you write yourself?**

I used Claude (Anthropic) to scaffold the initial notebook structure, including the `SelfAttentionModel` class, the training loops for both unmasked and masked models, the heatmap visualization code, and the LangChain prompt-engineering cells. I wrote/modified the prompt examples myself to make them my own, designed the specific math and logic problems, chose the prompting strategies, and authored all markdown explanations and the findings report. I also debugged and fixed three issues the assistant introduced (see below).

2. **Give one specific thing it produced that was wrong — a tensor shape error, a deprecated API, a loss function that trained but was wrong, a plausible-looking metric computed incorrectly. Paste the wrong output.**

The assistant generated code using deprecated/retired APIs in two places:

**Bug 1 — Deprecated LangChain import path:**
```python
from langchain.schema import HumanMessage, SystemMessage
```
```
ModuleNotFoundError: No module named 'langchain.schema'
```

**Bug 2 — Retired Gemini model name:**
```python
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", ...)
```
```
GoogleModelNotFoundError: Error calling model 'gemini-1.5-flash' (NOT_FOUND):
404 NOT_FOUND. models/gemini-1.5-flash is not found for API version v1beta
```

**Bug 3 — Triple-quote string collision:**
```python
4. "Jazz is more complex than pop music.""""
                                               ^
SyntaxError: unterminated string literal (detected at line 8)
```

3. **How did you find out? What did the failure look like?**

- **Bug 1:** `ModuleNotFoundError` immediately on cell execution. The `langchain` package was split into `langchain-core` in recent releases, moving schema classes to `langchain_core.messages`.
- **Bug 2:** A `404 NOT_FOUND` HTTP error from the Gemini API. Google retired the 1.5 model series; the current equivalent is `gemini-3.5-flash`.
- **Bug 3:** A `SyntaxError` at runtime. The string `music."` ended with a double quote that merged with the closing `"""`, creating four consecutive quotes which Python cannot parse.

4. **What did you change, and why does your version work?**

- **Bug 1 fix:** Changed `from langchain.schema import ...` → `from langchain_core.messages import HumanMessage, SystemMessage` and installed `langchain-core` instead of `langchain`. This matches the current package structure.
- **Bug 2 fix:** Changed `model="gemini-1.5-flash"` → `model="gemini-3.5-flash"`. This is a currently available model on the Gemini API.
- **Bug 3 fix:** Added a newline before the closing `"""` so the inner quote and the triple-quote delimiter are separated:
  ```python
  4. "Jazz is more complex than pop music."
  """
  ```
  Python now sees the inner `"` as part of the string content and `"""` as the delimiter.
