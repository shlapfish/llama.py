from functools import cache
from typing import Any

from llama import lib, ffi


def grammar_element(type, value: int) -> Any:
    ret = ffi.new("llama_grammar_element *")
    ret.dtype = type
    ret.value = value
    return ret


@cache
def ref(rule_nr: int) -> Any:
    return grammar_element(lib.LLAMA_GRETYPE_RULE_REF, rule_nr)


@cache
def char(c: str) -> Any:
    return grammar_element(lib.LLAMA_GRETYPE_CHAR, ord(c))


# end of rule definition
END = grammar_element(lib.LLAMA_GRETYPE_END, 0)
# start of alternate definition for rule
OR = grammar_element(lib.LLAMA_GRETYPE_ALT, 0)
CHAR_NOT = lib.LLAMA_GRETYPE_CHAR_NOT  # inverse char(s) ([^a], [^a-b] [^abc])

# modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to be an inclusive range ([a-z])
CHAR_RANGE = lib.LLAMA_GRETYPE_CHAR_RNG_UPPER

# modifies a preceding LLAMA_GRETYPE_CHAR or
# LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
CHAR_OR = lib.LLAMA_GRETYPE_CHAR_ALT
