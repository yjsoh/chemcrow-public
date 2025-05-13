from typing import Optional, List

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from ..logging_tool import PromptAndResponseLogger

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(model, temp, api_key, streaming: bool = False):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
        )
    elif model.startswith("meta-llama"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler(), PromptAndResponseLogger()],
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            cache=False,
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-4-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {},
        local_rxn: bool = False,
    ):
        """Initialize ChemCrow agent."""

        load_dotenv()
        try:
            self.llm = _make_llm(model, temp, openai_api_key, streaming)
        except ValidationError:
            raise ValueError("Invalid OpenAI API key")

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming)
            tools = make_tools(
                tools_llm, api_keys=api_keys, local_rxn=local_rxn, verbose=verbose
            )

        # Initialize callback handler
        self.callback_handler: List[BaseCallbackHandler] = [PromptAndResponseLogger()]
        # callback_manager: BaseCallbackManager = PromptAndResponseManager(
        #     handlers=self.callback_handler
        # )
        callback_manager = CallbackManager(handlers=self.callback_handler)

        # Initialize agent
        agent = ChatZeroShotAgent.from_llm_and_tools(
            self.llm,
            tools,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            question_prompt=QUESTION_PROMPT,
            # callback_manager=callback_manager,
            # callbacks=self.callback_handler,
        )

        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            # callback_manager=callback_manager,
            # callback_manager=None,
            verbose=True,
            max_iterations=max_iterations,
            # callbacks=self.callback_handler,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
