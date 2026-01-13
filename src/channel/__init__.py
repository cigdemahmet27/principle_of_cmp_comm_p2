"""
Channel package - Physical channel models.

Contains implementations for:
- Gilbert-Elliot burst error channel model
"""

from .gilbert_elliot import GilbertElliottChannel, ChannelState

__all__ = [
    'GilbertElliottChannel',
    'ChannelState'
]
