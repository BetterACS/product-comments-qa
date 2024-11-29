from typing import Optional

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM

from pocqa.pipeline.base_pipeline import POCQABasePipeline
from pocqa.types import QualityScore


class SimplePipeline(POCQABasePipeline):
    def __init__(self, llm: LLM):
        super(SimplePipeline, self).__init__(llm)
        self.judge = self.llm.as_structured_llm(output_cls=QualityScore, temperature=0.5)

    def assess(
        self,
        comment: str,
        informative_extraction_prompt: PromptTemplate = POCQABasePipeline.INFORMATION_EXTRACT_PROMPT,
        judge_prompt: PromptTemplate = POCQABasePipeline.JUDGE_PROMPT,
    ) -> QualityScore:
        response = self.llm.complete(informative_extraction_prompt.format(comment=comment))
        response = self.judge.complete(judge_prompt.format(comment=comment, detail=response.text))

        return QualityScore.model_validate_json(response.text)
