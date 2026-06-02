#################################################################################
# Copyright (c) 2018-2025, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################


from colorama import Fore


def add_color(string, color=None):
    if color:
        string = f'{color}{string}{Fore.RESET}'
    return string


def print_color(string, *args, **kwargs):
    if 'color' in kwargs:
        string = add_color(string, kwargs['color'])
        kwargs_copy = kwargs.copy()
        del kwargs_copy['color']
    else:
        kwargs_copy = kwargs
    print(string, *args, **kwargs_copy)


class PrintOnce:
    """Thread-safe print-once utility that tracks printed messages.

    This class provides a singleton-like interface for printing messages only once
    during program execution. It's useful for printing warnings or status messages
    that should not clutter the output.

    Example:
        >>> from quant_utils import PrintOnce
        >>> print_once = PrintOnce()
        >>> print_once("This prints once", color=Fore.YELLOW)
        >>> print_once("This prints once")  # Skipped
        >>> print_once.clear()  # Reset tracker
        >>> print_once("This prints once")  # Prints again
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._printed = {}
        return cls._instance

    def __call__(self, string, *args, **kwargs):
        """Print string only if it hasn't been printed before."""
        if string not in self._printed:
            print_color(string, *args, **kwargs)
            self._printed[string] = True

    def clear(self):
        """Clear all tracked messages, allowing them to be printed again."""
        self._printed.clear()

    def clear_message(self, string):
        """Clear a specific message from tracking."""
        self._printed.pop(string, None)

    def is_printed(self, string):
        """Check if a message has been printed."""
        return string in self._printed

    def set_messages(self, messages_dict):
        """Set/replace tracked messages (for disabling specific messages).

        Args:
            messages_dict: Dictionary where keys are message strings to track.
                          Useful for disabling certain messages.
        """
        self._printed = dict.fromkeys(messages_dict.keys(), True)

    @property
    def print_once_dict(self):
        """Backward compatibility property for direct dict access."""
        return self._printed

    @print_once_dict.setter
    def print_once_dict(self, value):
        """Backward compatibility setter for direct dict access."""
        self._printed = value if value is not None else {}


# Global instance for easy access
_print_once_instance = PrintOnce()
print_once = _print_once_instance
print_once_dict = _print_once_instance.print_once_dict  # Backward compatibility
