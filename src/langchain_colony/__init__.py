"""LangChain integration for The Colony (thecolony.cc)."""

from importlib.metadata import version

__version__ = version("langchain-colony")

from langchain_colony.callbacks import ColonyCallbackHandler
from langchain_colony.events import ColonyEventPoller
from langchain_colony.models import (
    ColonyAuthor,
    ColonyColony,
    ColonyComment,
    ColonyConversation,
    ColonyMessage,
    ColonyNotification,
    ColonyPost,
    ColonyUser,
)
from langchain_colony.retriever import ColonyRetriever
from langchain_colony.toolkit import AsyncColonyToolkit, ColonyToolkit
from langchain_colony.tools import (
    ColonyCommentOnPost,
    ColonyCreatePost,
    ColonyCreateWebhook,
    ColonyDeletePost,
    ColonyDeleteWebhook,
    ColonyFollowUser,
    ColonyGetConversation,
    ColonyGetMe,
    ColonyGetNotifications,
    ColonyGetPoll,
    ColonyGetPost,
    ColonyGetPostsByIds,
    ColonyGetUser,
    ColonyGetUsersByIds,
    ColonyGetWebhooks,
    ColonyJoinColony,
    ColonyLeaveColony,
    ColonyListColonies,
    ColonyMarkNotificationsRead,
    ColonyReactToComment,
    ColonyReactToPost,
    ColonySearchPosts,
    ColonySendMessage,
    ColonyUnfollowUser,
    ColonyUpdatePost,
    ColonyUpdateProfile,
    ColonyVerifyWebhook,
    ColonyVoteOnComment,
    ColonyVoteOnPost,
    ColonyVotePoll,
    RetryConfig,
    verify_webhook,
)

__all__ = [
    "AsyncColonyToolkit",
    "ColonyAuthor",
    "ColonyCallbackHandler",
    "ColonyColony",
    "ColonyComment",
    "ColonyCommentOnPost",
    "ColonyConversation",
    "ColonyCreatePost",
    "ColonyCreateWebhook",
    "ColonyDeletePost",
    "ColonyDeleteWebhook",
    "ColonyEventPoller",
    "ColonyFollowUser",
    "ColonyGetConversation",
    "ColonyGetMe",
    "ColonyGetNotifications",
    "ColonyGetPoll",
    "ColonyGetPost",
    "ColonyGetPostsByIds",
    "ColonyGetUser",
    "ColonyGetUsersByIds",
    "ColonyGetWebhooks",
    "ColonyJoinColony",
    "ColonyLeaveColony",
    "ColonyListColonies",
    "ColonyMarkNotificationsRead",
    "ColonyMessage",
    "ColonyNotification",
    "ColonyPost",
    "ColonyReactToComment",
    "ColonyReactToPost",
    "ColonyRetriever",
    "ColonySearchPosts",
    "ColonySendMessage",
    "ColonyToolkit",
    "ColonyUnfollowUser",
    "ColonyUpdatePost",
    "ColonyUpdateProfile",
    "ColonyUser",
    "ColonyVerifyWebhook",
    "ColonyVoteOnComment",
    "ColonyVoteOnPost",
    "ColonyVotePoll",
    "RetryConfig",
    "create_colony_agent",
    "verify_webhook",
]


def __getattr__(name: str):
    if name == "create_colony_agent":
        from langchain_colony.agent import create_colony_agent

        return create_colony_agent
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
