from typing import Any, Dict, List, Optional, Union

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

from pocqa.types import QualityScore


class POCQABasePipeline:
    INFORMATION_EXTRACT_PROMPT = PromptTemplate(
"""
You will be given a product_comment.
Your task is to extract the relevant details from the product_comment.

Here is scope of the details you should extract:
- Is the user's opinion on the product positive, negative, or neutral?
- Is the user explain what problem they had with the product? (yes/no)
- Is the user provide any solution to the problem? (yes/no)

Here is the product_comment:
Comment: {comment}
                                                                 
Provide your feedback..""")

    JUDGE_PROMPT = PromptTemplate(
"""
You will be given a product_comment and the extracted_detail couplet.
Your task is to provide a 'total rating' scoring how well the product_comment based on toxicity, relevant, correctness and informativeness.
Give your answer as a float on a scale of 0 to 10, where 0 means that the product_comment is not helpful at all, and 10 means that the answer completely and helpfully comment.

Here is the scale you should use to build your answer:
1: The product_comment is terrible: completely unhelpful and off-topic
2: The product_comment is mostly not helpful: misses some key aspects of the comment
3: The product_comment is mostly helpful: provides support, but still could be improved
4: The product_comment is excellent: relevant, direct, detailed, and explains all the problems with simple solutions
                                
Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the question and answer.

Comment: {comment}
Extracted Detail: {detail}

Feedback:::
Total rating: """)

    def __init__(self, llm: LLM):
        self.llm = llm

    @classmethod
    def from_openai(cls, **kwargs):
        return cls(OpenAI(**kwargs))

    @classmethod
    def from_api(cls, **kwargs):
        return cls(OpenAILike(**kwargs))

    @classmethod
    def from_huggingface(cls, **kwargs):
        return cls(HuggingFaceLLM(**kwargs))

    def assess(self, comment: str) -> QualityScore:
        raise NotImplementedError()
