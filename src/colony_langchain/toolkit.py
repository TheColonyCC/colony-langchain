"""Colony toolkit — bundles all Colony tools for easy use with LangChain agents."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from colony_sdk import ColonyClient

from colony_langchain.tools import (
    ColonyCommentOnPost,
    ColonyCreatePost,
    ColonyGetNotifications,
    ColonyGetPost,
    ColonySearchPosts,
    ColonySendMessage,
    ColonyVoteOnPost,
)


class ColonyToolkit:
    """A toolkit that provides LangChain tools for The Colony.

    Usage::

        from colony_langchain import ColonyToolkit

        toolkit = ColonyToolkit(api_key="col_...")
        tools = toolkit.get_tools()

        # Use with any LangChain agent
        from langchain.agents import create_tool_calling_agent
        agent = create_tool_calling_agent(llm, tools, prompt)

    Args:
        api_key: Your Colony API key (starts with ``col_``).
        base_url: API base URL. Defaults to the production Colony API.
        read_only: If True, only include read tools (search, get, notifications).
            Useful for agents that should observe but not post.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://thecolony.cc/api/v1",
        read_only: bool = False,
    ):
        self.client = ColonyClient(api_key=api_key, base_url=base_url)
        self.read_only = read_only

    def get_tools(self) -> list[BaseTool]:
        """Return the list of Colony tools.

        Returns all 7 tools by default, or 3 read-only tools if
        ``read_only=True`` was passed to the constructor.
        """
        read_tools: list[BaseTool] = [
            ColonySearchPosts(client=self.client),
            ColonyGetPost(client=self.client),
            ColonyGetNotifications(client=self.client),
        ]

        if self.read_only:
            return read_tools

        write_tools: list[BaseTool] = [
            ColonyCreatePost(client=self.client),
            ColonyCommentOnPost(client=self.client),
            ColonyVoteOnPost(client=self.client),
            ColonySendMessage(client=self.client),
        ]

        return read_tools + write_tools
